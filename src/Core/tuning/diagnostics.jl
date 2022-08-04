############################################################################################
"""
$(TYPEDEF)

SMC Diagnostics container, including diagnostics of kernels used in jittering step.

# Fields
$(TYPEDFIELDS)
"""
struct SMCDiagnostics{P,J,T<:AbstractFloat,G} <: AbstractDiagnostics
    "Diagnostics used for all Baytes kernels"
    base::BaytesCore.BaseDiagnostics{Vector{P}}
    "Weighted average of incremental log weights - can be used for marginal or log predictive likelihood computation."
    ℓincrement::Float64
    "Cumulative log weights. If resampled, updated with jittered parameter - as weights will be used for tempering."
    ℓweights::Vector{Float64}
    "Log weights at current iteration, used for resampling criterion. NOT updated with jittered theta parameter as ℓweightsₜ depends on t-1 and jittered theta parameter come in at next iteration."
    ℓweightsₜ::Vector{Float64}
    "Normalized Log weights. They might be adjusted from previous iterations, so will differ from ℓweights. Will be set to log(1/N) if resampling applied."
    ℓweightsₙ::Vector{Float64}
    "Diagnostics of jitter steps. If not resampled this iteration (i.e., resampled == false), contains jitterdiagnostics from previous step."
    jitterdiagnostics::Vector{J}
    "Number of jittering steps"
    jittersteps::Int64
    "Correlation from rejuvented continuous parameter"
    ρ::Vector{T}
    "ESS and accepted steps for SMC kernel"
    ESS::Float64
    "Boolean if step has been resampled."
    resampled::Bool
    "Generated quantities specified for objective"
    generated::G
    function SMCDiagnostics(
        base::BaytesCore.BaseDiagnostics{Vector{P}},
        ℓincrement::Float64,
        ℓweights::Vector{Float64},
        ℓweightsₜ::Vector{Float64},
        ℓweightsₙ::Vector{Float64},
        jitterdiagnostics::Vector{J},
        jittersteps::Int64,
        ρ::Vector{T},
        ESS::Float64,
        resampled::Bool,
        generated::G
    ) where {P,J,T<:AbstractFloat,G}
        return new{P,J,T,G}(
            base,
            ℓincrement,
            ℓweights,
            ℓweightsₜ,
            ℓweightsₙ,
            jitterdiagnostics,
            jittersteps,
            ρ,
            ESS,
            resampled,
            generated
        )
    end
end

############################################################################################
function generate_showvalues(diagnostics::D) where {D<:SMCDiagnostics}
    return function showvalues()
        return (:smc, "diagnostics"),
        (:iter, diagnostics.base.iter),
        (:Avgℓobjective, mean(diagnostics.ℓweights)),
        (:Temperature, diagnostics.base.temperature),
        (:ESS, diagnostics.ESS),
        (:resampled, diagnostics.resampled),
        (:AvgJitterCorrelation, mean(diagnostics.ρ))
    end
end

############################################################################################
#export
export SMCDiagnostics, generate_showvalues
