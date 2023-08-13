############################################################################################
"""
$(TYPEDEF)

Stores buffers for computing correlation of model parameter particles.

# Fields
$(TYPEDFIELDS)
"""
struct CorrelationBuffer{T<:AbstractFloat}
    "Correlation between old and rejuvenated parameter"
    ρ::Vector{T}
    "Pairs of old and new parameter samples - for the same parameter"
    pairs::Vector{Vector{T}}
    function CorrelationBuffer(
        Nparameter::I,
        Nchains::I,
        F::Type{T}
    ) where {I<:Integer, T<:AbstractFloat}
        ρ = zeros(T, Nparameter)
        pairs = map(iter -> zeros(T, Nchains), Base.OneTo(2))
        return new{T}(ρ, pairs)
    end
end
function update!(buffer::CorrelationBuffer)
    @inbounds for iter in eachindex(buffer.ρ)
        buffer.ρ[iter] = 0.0
    end
    return nothing
end

"""
$(SIGNATURES)
Compute inplace the correlation between old parameter vectors `valᵤ` and new parameter vectors from 'algorithmᵛ'.

# Examples
```julia
```

"""
function compute_ρ!(
    ρ::Vector{T},
    pairs::Vector{Vector{F}},
    valᵤ::Vector{Vector{S}},
    algorithmᵛ,
    ϵ = 1e-10
) where {S<:Real, T<:AbstractFloat,F<:Real}
    for iter in eachindex(ρ)
        for chain in eachindex(valᵤ)
            pairs[1][chain] = valᵤ[chain][iter]
            pairs[2][chain] = BaytesCore.get_result(algorithmᵛ[chain]).θᵤ[iter]
        end
        #Assign a small value at start of index for temporary starting point pair[1]
        #!NOTE: If pairs[i] are constant (i.e., all resampled initial parameter come from same index), ρ = NaN
        #!NOTE: Will not change parameter as scalar immutable and only buffer changed
        #!NOTE: Otherwise fun(ρ) = NaN, which causes jittering until max jittering criterion is met.
        pairs[1][begin] += ϵ
        ρ[iter] = Statistics.cor(pairs[1], pairs[2])
    end
    return ρ
end
function compute_ρ!(buffer::CorrelationBuffer, valᵤ::Vector{Vector{S}}, algorithmᵛ) where {S<:Real}
    return compute_ρ!(buffer.ρ, buffer.pairs, valᵤ, algorithmᵛ)
end

############################################################################################
"""
$(TYPEDEF)

Buffer struct to inplace store intermediate values.

# Fields
$(TYPEDFIELDS)
"""
struct SMCBuffer{M<:NamedTuple,I<:Integer,L<:Real,T<:AbstractFloat,F<:AbstractFloat,P,J}
    "Buffer for cumulative particle weights at each iteration, such that they can be used again next iteration after resampling step."
    cumweights::Vector{T}
    "Buffer for model parameter and reweighting statistics."
    parameter::BaytesCore.ModelParameterBuffer{M, L, T, I}
    "Buffer to compute correlation between original and jittered parameter."
    correlation::CorrelationBuffer{F}
    "Predictions"
    predictions::Vector{P}
    "Diagnostics of jitter steps. If not jittered this iteration (i.e., accepted == true), contains jitterdiagnostics from previous step."
    jitterdiagnostics::Vector{J}
    function SMCBuffer(
        _rng::Random.AbstractRNG,
        kernel::K,
        model::M,
        data::D,
        Nchains::Int64;
        ancestortype::Type{I}=Int64,
    ) where {K<:AbstractAlgorithm,M<:ModelWrapper,D,I<:Integer}
#    ) where {K<:Union{MCMC,PMCMC},M<:ModelWrapper,D,I<:Integer}
        @argcheck 0 < Nchains "Nchains need to be positive"
        result = BaytesCore.get_result(kernel)
        Nparams = length(result.θᵤ)
        # Assign weights buffer for particle
        cumweights = zeros(model.info.reconstruct.default.output, Nchains)
        # Assign weights, buffer for parameter and correlation
        parameter = BaytesCore.ModelParameterBuffer(model, Nchains, Nparams, model.info.reconstruct.default.output, ancestortype)
        correlation = CorrelationBuffer(Nparams, Nchains, model.info.reconstruct.default.output)
        # Assign buffer for predictions
        TPrediction = BaytesCore.infer(_rng, kernel, model, data)
        predictions = Vector{TPrediction}(undef, Nchains)
        # Assign buffer for propagation and jitter diagnostics
        TJitter = BaytesCore.infer(_rng, AbstractDiagnostics, kernel, model, data)
        jitterdiagnostics = Vector{TJitter}(undef, Nchains)
        return new{
            typeof(model.val),
            eltype(parameter.index),
            eltype(result.θᵤ),
            eltype(parameter.weight),
            eltype(correlation.ρ),
            TPrediction,
            TJitter,
        }(
            cumweights,
            parameter,
            correlation,
            predictions,
            jitterdiagnostics,
        )
    end
end

function update!(buffer::SMCBuffer)
    update!(buffer.correlation)
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Compute maximum number for batchsize in resampling step of SMC.

# Examples
```julia
```

"""
function compute_batchsize(Nchains::Integer, Nthreads=Threads.nthreads()) #Polyester.num_threads()
    if Nthreads >= Nchains
        batchsize = 1
    elseif Nthreads < Nchains && mod(Nchains, Nthreads) == 0
        batchsize = Int64(Nchains / Nthreads)
    elseif Nthreads < Nchains && mod(Nchains, Nthreads) != 0
        println(
            "Warning: Number of θ particles is no multiple of available number of threads."
        )
        batchsize = Int64(ceil(Nchains / Nthreads))
    else
        println(
            "Could not determine optimal batchsize, use θ particles as multiple of available cores.",
        )
    end
    return batchsize
end

############################################################################################
# Export
export SMCBuffer, update!, compute_ρ
