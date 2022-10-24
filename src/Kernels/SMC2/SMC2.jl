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
    sym1 = get_sym(constructor.jitter)
    sym2 = get_sym(constructor.propagation)
    #Return unique symbols as tuple
    sym = unique((sym1..., sym2...))
    return Tuple(sym)
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

function (constructor::SMC2Constructor)(
    _rng::Random.AbstractRNG,
    model::ModelWrapper,
    data::D,
    proposaltune::P,
    info::BaytesCore.SampleDefault
) where {D, P<:BaytesCore.ProposalTune}
    @unpack propagation, jitter = constructor
    # Start with jitter kernel in case model parameter are sampled from prior, then initialize propagation kernel
    _jitter = jitter(_rng, model, data, proposaltune, info)
    _propagation = propagation(_rng, model, data, proposaltune, info)
    return SMC2Kernel(_propagation, _jitter)
end

function result!(kernel::SMC2Kernel, result::L) where {L<:ℓObjectiveResult}
    result!(kernel.pmcmc, result)
    return nothing
end

function get_result(kernel::SMC2Kernel)
    return get_result(kernel.pmcmc)
end
function predict(_rng::Random.AbstractRNG, kernel::SMC2Kernel, objective::Objective)
    return predict(_rng, kernel.pf, objective)
end

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

function SMCweight(_rng::Random.AbstractRNG, algorithm::SMC2Kernel, objective::Objective, proposaltune::P, cumweightsₜ₋₁) where {P<:ProposalTune}
    #!NOTE: Use cumulative weight from propagated pf, taking into account new data
    cumweightsₜ = objective.temperature * algorithm.pf.particles.ℓobjective.cumulative #get_ℓweight(algorithm.pf)
    #!NOTE: cumweightsₜ₋₁ already has correct temperature at t-1
    return cumweightsₜ, cumweightsₜ - cumweightsₜ₋₁
end
function SMCreweight(_rng::Random.AbstractRNG, algorithm::SMC2Kernel, objective::Objective, proposaltune::P, cumweightsₜ₋₁) where {P<:ProposalTune}
    #!NOTE: Run Particle Filter wrt to jittered model parameter, and compute new weight
    #!NOTE: Update PF particles to PMCMC.PF particles in this step, so UpdateTrue()
    proposaltune_updatetrue = BaytesCore.ProposalTune(objective.temperature, BaytesCore.UpdateTrue(), proposaltune.datatune)
    propose!(
        _rng,
        algorithm.pf,
        objective.model,
        objective.data,
        proposaltune_updatetrue
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
#    @unpack propagation, jitter = kernel
    ## Initialize individual PF and PMCMC algorithm
    @unpack model, data, temperature = objective
    @unpack Ntuning = tune
    Nchains = tune.chains.Nchains
    modelᵗᵉᵐᵖ = deepcopy(model)
    tagged = Tagged(model, kernel.propagation.sym)
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
    #!NOTE: Train with fully available data
    proposaltune = BaytesCore.ProposalTune(temperature, BaytesCore.UpdateTrue(), BaytesCore.DataTune(BaytesCore.Batch()))
    proposaltune_captured = BaytesCore.ProposalTune(temperature, tune.capture, BaytesCore.DataTune(BaytesCore.Batch()))
    algorithmᵛ = [kernel(_rng, modelᵛ[iter], data, proposaltune_captured, info) for iter in Base.OneTo(Nchains)]

    ## Assign weights
    weights = BaytesCore.ParameterWeights(Nchains)
    ## Assign buffer for inplace resampling
    buffer = SMCBuffer(
        _rng, algorithmᵛ[1].pmcmc, modelᵗᵉᵐᵖ, data, Nchains
    )
    ## Loop through all models
    #!NOTE: Polyester may change type to StridedArray, which is not supported in SMC kernel INITIATION
#    Polyester.@batch per=thread minbatch=tune.batchsize for iter in eachindex(algorithmᵛ)
    Base.Threads.@threads for iter in eachindex(algorithmᵛ)
        ## Tune PMCMC algorithm - always with UpdateTrue for first iteration
        propose!(_rng, algorithmᵛ[iter].pmcmc, modelᵛ[iter], data, proposaltune)
        for _ in 2:Ntuning
            propose!(_rng, algorithmᵛ[iter].pmcmc, modelᵛ[iter], data, proposaltune_captured)
        end
        ## Assign a Particle Filter for tuned theta
        propose!(_rng, algorithmᵛ[iter].pf, modelᵛ[iter], data, proposaltune)
        ## Calculate initial weights
        _objective = Objective(modelᵛ[iter], objective.data, tune.tagged, objective.temperature)
        buffer.cumweights[iter], _ = SMCweight( #weights.buffer[iter] = SMCweight(
            _rng,
            algorithmᵛ[iter],
            _objective,
            proposaltune_captured,
            _objective(_objective.model.val)
        )
    end
    ## Normalize weights
    #!NOTE: Skip this step and leave initial particles equal weighted - so no resampling at first iteration!
#    weights(weights.buffer)
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
function propagate!(_rng::Random.AbstractRNG, particles::SMCParticles{<:SMC2Kernel}, tune::SMCTune, data::D, proposaltune::P) where {D, P<:ProposalTune}
    @unpack temperature = proposaltune
    ## Propagate series forward with recent particle
#    Polyester.@batch per=thread minbatch=tune.batchsize for iter in eachindex(particles.model)
    #!NOTE: Always updated in case resampling has been applied or dynamics depend on data dimension
    proposaltune_updated = BaytesCore.ProposalTune(proposaltune.temperature, BaytesCore.UpdateTrue(), proposaltune.datatune)
    Base.Threads.@threads for iter in eachindex(particles.model)
        #_, diagnostics = propagate!(
        propagate!(
            _rng,
            particles.kernel[iter].pf,
            particles.model[iter],
            data,
            proposaltune_updated
        )
        #particles.buffer.predictions[iter] = diagnostics.base.prediction
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
