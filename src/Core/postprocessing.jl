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
    proposaltune::P,
    info::BaytesCore.SampleDefault
) where {D, P<:BaytesCore.ProposalTune}
    return SMC(
        _rng,
        constructor.kernel,
        Objective(model, data, Tagged(model), proposaltune.temperature),
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
    if t > 1
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
    end
    # Get indices where jittering has been applied and all correlation numbers are non-NaN for computing output statistics.
    #    rejuvenations = [diagnosticsᵛ[iter].resampled for iter in eachindex(diagnosticsᵛ)]
    rejuvenations = [diagnosticsᵛ[iter].resampled && all(!isnan, diagnosticsᵛ[iter].ρ) for iter in eachindex(diagnosticsᵛ)]
    if sum(rejuvenations) > 0
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
    end
    ## ESS quantiles
    println(
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
    ## Return
    return nothing
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

    #Depends if jitter diagnostics should be recorded
    if isa(smc.tune.jitterdiagnostics, BaytesCore.UpdateTrue)
        TJitter = BaytesCore.infer(_rng, AbstractDiagnostics, smc.particles.kernel[begin], model, data)
    else
        TJitter = Nothing
    end
    TTemperature = model.info.reconstruct.default.output
    TGenerated, TGenerated_algorithm = infer_generated(_rng, smc, model, data)
    return SMCDiagnostics{TPrediction,TJitter,TTemperature,TGenerated, TGenerated_algorithm}
end

############################################################################################
"""
$(SIGNATURES)
Infer type of generated quantities of PF sampler.

# Examples
```julia
```

"""
function infer_generated(
    _rng::Random.AbstractRNG, smc::SMC, model::ModelWrapper, data::D
) where {D}
    objective = Objective(model, data, smc.tune.tagged)
    TGenerated = typeof(generate(_rng, objective, smc.tune.generated))
    TGenerated_algorithm = typeof(generate(_rng, smc, objective, smc.tune.generated))
    return TGenerated, TGenerated_algorithm
end

############################################################################################
"""
$(SIGNATURES)
Generate statistics for algorithm given model parameter and data.

# Examples
```julia
```

"""
function generate(_rng::Random.AbstractRNG, algorithm::SMC, objective::Objective)
    return nothing
end
function generate(_rng::Random.AbstractRNG, algorithm::SMC, objective::Objective, gen::BaytesCore.UpdateTrue)
    return generate(_rng, algorithm, objective)
end
function generate(_rng::Random.AbstractRNG, algorithm::SMC, objective::Objective, gen::BaytesCore.UpdateFalse)
    return nothing
end

############################################################################################
#export
export SMCConstructor, infer
