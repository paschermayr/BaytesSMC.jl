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
    "Weighted average of incremental log weights - can be used for marginal likelihood computation."
    ℓincrement::Float64
    "Log weights for resampling steps."
    ℓweights::Vector{Float64}
    "Normalized Log weights."
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
