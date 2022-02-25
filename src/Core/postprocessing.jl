############################################################################################
"""
$(SIGNATURES)
Callable struct to make initializing SMC sampler easier in sampling library.

# Examples
```julia
```

"""
struct SMCConstructor{J<:AbstractConstructor,D<:SMCDefault} <: AbstractConstructor
    "Jitter kernel"
    kernel::J
    "SMC Default Arguments"
    default::D
    function SMCConstructor(
        kernel::J, default::D
    ) where {J<:AbstractConstructor,D<:SMCDefault}
        return new{J,D}(kernel, default)
    end
end
function (constructor::SMCConstructor)(
    _rng::Random.AbstractRNG,
    model::ModelWrapper,
    data::D,
    temperature::F,
    info::BaytesCore.SampleDefault
) where {D, F<:AbstractFloat}
    return SMC(
        _rng,
        constructor.kernel,
        Objective(model, data, Tagged(model), temperature),
        constructor.default,
        info
    )
end
function SMC(
    kernel::J; kwargs...
) where {J<:AbstractConstructor}
    return SMCConstructor(kernel, SMCDefault(; kwargs...))
end

function get_sym(constructor::SMCConstructor)
    return get_sym(constructor.kernel)
end

############################################################################################
"""
$(SIGNATURES)
Print diagnostics result of SMC sampler.

# Examples
```julia
```

"""
function results(
    diagnosticsᵛ::AbstractVector{M}, smc::SMC, Ndigits::Int64, quantiles::Vector{T}
) where {T<:Real,M<:SMCDiagnostics}
    ## Print Trace
    println("SMC algorithm with jitter kernel ", Base.nameof(typeof(smc.particles.kernel[begin])), " Diagnostics:")
    println(
        "Sampler finished after ",
        size(diagnosticsᵛ, 1),
        " iterations with average acceptance ratio of ",
        round(
            Statistics.mean(
                !diagnosticsᵛ[iter].resampled for iter in eachindex(diagnosticsᵛ)
            ) * 100;
            digits=Ndigits,
        ),
        "%.",
    )
    println(
        "Initial average ℓlikelihood per particle: ",
        round(Statistics.mean(diagnosticsᵛ[begin].ℓweights); digits=Ndigits),
        ", variance: ",
        round(Statistics.var(diagnosticsᵛ[begin].ℓweights); digits=Ndigits),
    )
    println(
        "Final average ℓlikelihood per particle: ",
        round(Statistics.mean(diagnosticsᵛ[end].ℓweights); digits=Ndigits),
        ", variance: ",
        round(Statistics.var(diagnosticsᵛ[end].ℓweights); digits=Ndigits),
    )
    ## Number of jittering steps
    println("Total number of jittering steps: ",
        Statistics.sum(diagnosticsᵛ[iter].jittersteps for iter in eachindex(diagnosticsᵛ)),
        ".")
    ## Rejuvenation steps
    t = convert(Int64, floor(length(diagnosticsᵛ) / 2))
    println(
        "Rejuvenations on average every ",
        round(
            (
                1 / (
                    1 - Statistics.mean(
                        !diagnosticsᵛ[iter].resampled for iter in eachindex(diagnosticsᵛ)
                    )
                )
            );
            digits=Ndigits,
        ),
        " step, every ",
        round(
            1 / (1 - Statistics.mean(!diagnosticsᵛ[iter].resampled for iter in 1:t));
            digits=Ndigits,
        ),
        " steps in first half, every ",
        round(
            1 / (
                1 - Statistics.mean(
                    !diagnosticsᵛ[iter].resampled for iter in (t + 1):length(diagnosticsᵛ)
                )
            );
            digits=Ndigits,
        ),
        " steps in second half.",
    )
    ##Rejuvenation Correlation
    rejuvenations = [diagnosticsᵛ[iter].resampled for iter in eachindex(diagnosticsᵛ)]
    println(
        "Quantiles for average rejuvenation correlation of parameter: ",
        round.(
            Statistics.quantile(
                Statistics.mean.(
                    #!NOTE: If pairs[i] are constant (i.e., all resampled initial parameter come from same index), ρ = NaN for index
                    #filter(!isnan, diagnosticsᵛ[rejuvenations][iter].ρ) for
                    diagnosticsᵛ[rejuvenations][iter].ρ for iter in Base.OneTo(sum(rejuvenations))
                ),
                quantiles,
            );
            digits=Ndigits,
        ),
        ".",
    )
    println(
        "Quantiles for average number rejuvenation steps: ",
        round.(
            Statistics.quantile(
                Statistics.mean.(
                    diagnosticsᵛ[rejuvenations][iter].jittersteps for
                    iter in Base.OneTo(sum(rejuvenations))
                ),
                quantiles,
            );
            digits=Ndigits,
        ),
        ".",
    )
    ## ESS quantiles
    return println(
        "Quantiles for ESS: ",
        round.(
            Statistics.quantile(
                [diagnosticsᵛ[iter].ESS for iter in eachindex(diagnosticsᵛ)], quantiles
            );
            digits=Ndigits,
        ),
        " for ",
        smc.tune.chains.Nchains,
        " chains.",
    )
end

############################################################################################
"""
$(SIGNATURES)
Infer MCMC diagnostics type.

# Examples
```julia
```

"""
function infer(
    _rng::Random.AbstractRNG,
    diagnostics::Type{AbstractDiagnostics},
    smc::SMC,
    model::ModelWrapper,
    data::D,
) where {D}
    #Pick one of the kernels to obtain type for prediction and jittering diagnostics.
    TPrediction = BaytesCore.infer(_rng, smc.particles.kernel[begin], model, data)
    TJitter = BaytesCore.infer(_rng, AbstractDiagnostics, smc.particles.kernel[begin], model, data)
    TTemperature = model.info.flattendefault.output
    return SMCDiagnostics{TPrediction,TJitter,TTemperature}
end

############################################################################################
#export
export SMCConstructor, infer
