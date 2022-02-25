############################################################################################
struct SMC2Constructor{W<:AbstractConstructor,J<:AbstractConstructor} <:AbstractConstructor
    "Constructor for (online) state propagation over time."
    propagation::W
    "Constructor for jittering step."
    jitter::J
    function SMC2Constructor(propagation::W, jitter::J) where {W<:AbstractConstructor,J<:AbstractConstructor}
        return new{W,J}(propagation, jitter)
    end
end
function get_sym(constructor::SMC2Constructor)
    return get_sym(constructor.jitter)
end

function SMC2(
    propagation::J, jitter::W; kwargs...
) where {J<:AbstractConstructor, W<:AbstractConstructor}
    return SMC(SMC2Constructor(propagation, jitter); kwargs...)
end

############################################################################################
"""
$(TYPEDEF)

SMC2 Kernel, consisting of a kernel for the particle trajectory and kernel for all model parameter.

# Fields
$(TYPEDFIELDS)
"""
struct SMC2Kernel{P<:ParticleFilter,M<:PMCMC} <: BaytesCore.AbstractAlgorithm
    "Kernel for (online) state propagation over time."
    pf::P   #θ dimension
    "Kernel for jittering step."
    pmcmc::M   #particle dimension
    function SMC2Kernel(pf::P, pmcmc::M) where {P<:ParticleFilter,M<:PMCMC}
        @argcheck isa(pf.tune.referencing, Marginal) "Cannot use Conditional Filter in θ dimension, use Marginal referencing instead"
        return new{P,M}(pf, pmcmc)
    end
end

function result!(kernel::SMC2Kernel, result::L) where {L<:ℓObjectiveResult}
    result!(kernel.pmcmc, result)
    return nothing
end

function get_result(kernel::SMC2Kernel)
    return get_result(kernel.pmcmc)
end
#=
function get_ℓweight(kernel::SMC2Kernel)
    return get_ℓweight(kernel.pmcmc)
end
=#
function infer(
    _rng::Random.AbstractRNG, kernel::SMC2Kernel, model::ModelWrapper, data::D
) where {D}
    return BaytesCore.infer(_rng, kernel.pf, model, data)
end
function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    kernel::SMC2Kernel,
    model::ModelWrapper,
    data::D,
) where {D}
    return BaytesCore.infer(_rng, diagnostics, kernel.pmcmc, model, data)
end

############################################################################################

function jitterkernel(particles::SMCParticles{<:SMC2Kernel}, iter::Int64)
    return particles.kernel[iter].pmcmc
end

function SMCweight(_rng::Random.AbstractRNG, objective::Objective, algorithm::SMC2Kernel, cumweightsₜ₋₁)
    #!NOTE: Use cumulative weight from propagated pf, taking into account new data
    cumweightsₜ = objective.temperature * algorithm.pf.particles.ℓobjective.cumulative #get_ℓweight(algorithm.pf)
    #!NOTE: cumweightsₜ₋₁ already has correct temperature at t-1
    return cumweightsₜ, cumweightsₜ - cumweightsₜ₋₁
end
function SMCreweight(_rng::Random.AbstractRNG, objective::Objective, algorithm::SMC2Kernel, cumweightsₜ₋₁)
    #!NOTE: Run Particle Filter wrt to jittered model parameter, and compute new weight
    propose!(
        _rng,
        algorithm.pf,
        objective.model,
        objective.data,
        objective.temperature,
        BaytesCore.UpdateTrue()
    )
    cumweightsₜ = objective.temperature * algorithm.pf.particles.ℓobjective.cumulative #get_ℓweight(algorithm.pf)
    #!NOTE: cumweightsₜ₋₁ already has correct temperature at t-1
    return cumweightsₜ, cumweightsₜ - cumweightsₜ₋₁
end

############################################################################################
"""
$(SIGNATURES)
Initialize SMC² sampler.

# Examples
```julia
```

"""
function SMCParticles(
    _rng::Random.AbstractRNG,
    kernel::SMC2Constructor,
    objective::Objective,  #Initial Model with prior (Parameter values are overwritten by prior samples)
    info::BaytesCore.SampleDefault,
    tune::SMCTune
)
    @unpack propagation, jitter = kernel
    ## Initialize individual PF and PMCMC algorithm
    @unpack model, data, temperature = objective
    @unpack Ntuning = tune
    Nchains = tune.chains.Nchains
    modelᵗᵉᵐᵖ = deepcopy(model)
    tagged = Tagged(model, propagation.sym)
    ModelWrappers.fill!(
        modelᵗᵉᵐᵖ,
        tagged,
        BaytesCore.to_NamedTuple(
            keys(tagged.parameter),
            getfield(modelᵗᵉᵐᵖ.val, keys(tagged.parameter)[1])[1:maximum(size(data))],
        ),
    )
    ## Initialize individual Models ~ Assign copies for  Model and Algorithms ~ necessary in initiation so no pointer error
    modelᵛ = [deepcopy(modelᵗᵉᵐᵖ) for _ in Base.OneTo(Nchains)]
    algorithmᵛ = [
        SMC2Kernel(
            propagation(_rng, modelᵛ[iter], data, temperature, info),
            jitter(_rng, modelᵛ[iter], data, temperature, info),
        ) for iter in Base.OneTo(Nchains)
    ]
    ## Assign weights
    weights = BaytesCore.ParameterWeights(Nchains)
    ## Assign buffer for inplace resampling
    buffer = SMCBuffer(
        _rng, algorithmᵛ[1].pmcmc, modelᵗᵉᵐᵖ, data, Nchains
    )
    ## Loop through all models
    #Polyester.@batch per=thread minbatch=tune.batchsize for iter in eachindex(algorithmᵛ)
    Base.Threads.@threads for iter in eachindex(algorithmᵛ)
        ## Tune PMCMC algorithm
        propose!(_rng, algorithmᵛ[iter].pmcmc, modelᵛ[iter], data, temperature, BaytesCore.UpdateTrue())
        for _ in 2:Ntuning
            propose!(_rng, algorithmᵛ[iter].pmcmc, modelᵛ[iter], data, temperature, tune.capture)
        end
        ## Assign a Particle Filter for tuned theta
        propose!(_rng, algorithmᵛ[iter].pf, modelᵛ[iter], data, temperature, BaytesCore.UpdateTrue())
        ## Calculate initial weights
        _objective = Objective(modelᵛ[iter], objective.data, tune.tagged, objective.temperature)
        buffer.cumweights[iter], weights.buffer[iter] = SMCweight(
            _rng,
            _objective,
            algorithmᵛ[iter],
            _objective(_objective.model.val)
        )
    end
    ## Normalize weights
    weights(weights.buffer)
    ## Return container
    return SMCParticles(modelᵛ, algorithmᵛ, weights, buffer)
end

############################################################################################
"""
$(SIGNATURES)
Propagate data forward over time.

# Examples
```julia
```

"""
function propagate!(_rng::Random.AbstractRNG, particles::SMCParticles{<:SMC2Kernel}, tune::SMCTune, data::D, temperature::F) where {D, F<:AbstractFloat}
    #!NOTE: Here, smc.buffer.jitterdiagnostics will not be updated, and for SMCDiagnostics, last available jitterdiagnostics are provided.
    ## Propagate series forward with recent particle
    #Polyester.@batch per=thread minbatch=tune.batchsize for iter in eachindex(particles.model)
    Base.Threads.@threads for iter in eachindex(particles.model)
        _, diagnostics = propagate!(
            _rng,
            particles.kernel[iter].pf,
            particles.model[iter],
            data,
            temperature
        )
        particles.buffer.predictions[iter] = diagnostics.base.prediction
    end
    return nothing
end

############################################################################################
#export
export
    SMC2,
    SMC2Constructor,
    SMC2Kernel,
    SMCweight,
    resample!,
    propagate!,
    weight!
