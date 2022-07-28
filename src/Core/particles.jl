############################################################################################
"""
$(TYPEDEF)

SMC kernel containing particles, dynamics and buffers for sampling.

# Fields
$(TYPEDFIELDS)
"""
struct SMCParticles{B<:AbstractAlgorithm,A<:ModelWrapper,D<:SMCBuffer}
    "SMC particles used for sampling."
    model::Vector{A}
    "Individual kernels associated to an SMC particle."
    kernel::Vector{B}
    "Weights struct for all SMC particles."
    weights::BaytesCore.ParameterWeights
    "Buffer values for allocation free evaluations."
    buffer::D
    function SMCParticles(
        model::Vector{A},
        algorithm::Vector{B},
        weights::BaytesCore.ParameterWeights,
        buffer::D,
    ) where {B<:AbstractAlgorithm,A<:ModelWrapper,D<:SMCBuffer}
        return new{B,A,D}(model, algorithm, weights, buffer)
    end
end

############################################################################################
"""
$(SIGNATURES)
Initialize SMC particles.

# Examples
```julia
```

"""
function SMCParticles(
    _rng::Random.AbstractRNG,
    JitterKernel::AbstractConstructor,
    objective::Objective,
    info::BaytesCore.SampleDefault,
    tune::SMCTune
)
    ## Initialize individual MCMC algorithm
    @unpack data, temperature = objective
    @unpack tagged, Ntuning = tune
    Nchains = tune.chains.Nchains
    ## Adjust observed data to current knowledge
    modelᵗᵉᵐᵖ = deepcopy(objective.model)
    ## Initialize individual Models ~ Assign copies for  Model and Algorithms - info and id can be shared ~ necessary in initiation so no pointer issues
    modelᵛ = [
        ModelWrapper(deepcopy(modelᵗᵉᵐᵖ.val), deepcopy(modelᵗᵉᵐᵖ.arg), modelᵗᵉᵐᵖ.info, modelᵗᵉᵐᵖ.id) for
        _ in Base.OneTo(Nchains)
    ]
    algorithmᵛ = [JitterKernel(_rng, modelᵛ[iter], data, temperature, info) for iter in Base.OneTo(Nchains)]
    ## Assign weights
    weights = BaytesCore.ParameterWeights(Nchains)
    ## Assign buffer for inplace resampling
    buffer = SMCBuffer(
        _rng, algorithmᵛ[1], modelᵗᵉᵐᵖ, data, Nchains
    )
    ## Loop through all models
    #Polyester.@batch per=thread minbatch=tune.batchsize
    Base.Threads.@threads for iter in eachindex(algorithmᵛ)
        ## Tune kernel
        #Update kernel for first iteration
        propose!(_rng, algorithmᵛ[iter], modelᵛ[iter], data, temperature, BaytesCore.UpdateTrue())
        for _ in 2:Ntuning
            propose!(_rng, algorithmᵛ[iter], modelᵛ[iter], data, temperature, tune.capture)
        end
        ## Assign weight
        #!NOTE: Objective itself only stores logposterior function, no problem to call it as long as θ comes from modelᵛ.
        _objective = Objective(modelᵛ[iter], objective.data, tune.tagged, objective.temperature)
        buffer.cumweights[iter], weights.buffer[iter] = SMCweight(
            _rng,
            _objective,
            algorithmᵛ[iter],
            _objective.temperature * _objective(objective.model.val)
        )
    end
    ## Normalize weights
    weights(weights.buffer)
    ## Return container
    return SMCParticles(modelᵛ, algorithmᵛ, weights, buffer)
end

############################################################################################
function SMCDiagnostics(
    smc::SMCParticles, temperature::T, ESS::Float64, accepted::Bool, jittersteps::Int64, iter::Int64, generated::G, jitterdiag::UpdateTrue
) where {T<:AbstractFloat, G}
    return SMCDiagnostics(
        BaytesCore.BaseDiagnostics(
            Statistics.mean(smc.buffer.cumweights),
            temperature,
            copy(smc.buffer.predictions),
            iter-1
        ),
        BaytesCore.weightedincrement(smc.weights),
        #!NOTE: There is not really any way around making a copy from buffer if no pointer issue for diagnostics
        copy(smc.buffer.cumweights),
        copy(smc.weights.ℓweightsₙ),
        deepcopy(smc.buffer.jitterdiagnostics),
        jittersteps,
        copy(smc.buffer.correlation.ρ),
        ESS,
        accepted,
        generated
    )
end
function SMCDiagnostics(
    smc::SMCParticles, temperature::T, ESS::Float64, accepted::Bool, jittersteps::Int64, iter::Int64, generated::G, jitterdiag::UpdateFalse
) where {T<:AbstractFloat, G}
    return SMCDiagnostics(
        BaytesCore.BaseDiagnostics(
            Statistics.mean(smc.buffer.cumweights),
            temperature,
            copy(smc.buffer.predictions),
            iter-1
        ),
        BaytesCore.weightedincrement(smc.weights),
        #!NOTE: There is not really any way around making a copy from buffer if no pointer issue for diagnostics
        copy(smc.buffer.cumweights),
        copy(smc.weights.ℓweightsₙ),
        [nothing],
        jittersteps,
        copy(smc.buffer.correlation.ρ),
        ESS,
        accepted,
        generated
    )
end

############################################################################################
#export
export SMCParticles, init, resample!, propagate!, weight!
