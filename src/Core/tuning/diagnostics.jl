############################################################################################
"""
$(TYPEDEF)

SMC Diagnostics container, including diagnostics of kernels used in jittering step.

# Fields
$(TYPEDFIELDS)
"""
struct SMCDiagnostics{P,J,T<:AbstractFloat} <: AbstractDiagnostics
    "Log likelihood of individual Particle Filter/MCMC call. Recorded to perform model selection via marginal likelihood."
    ℓℒ::Vector{Float64}
    "Temperature to perform proposal steps and propagation"
    temperature::T
    "Predictions. Note that this may differ from predicitions in jitterdiagnostics, as prediction field is updated each iteration."
    prediction::Vector{P}
    "Diagnostics of jitter steps. If not resampled this iteration (i.e., resampled == false), contains jitterdiagnostics from previous step."
    jitterdiagnostics::Vector{J}
    "Number of jittering steps"
    jittersteps::Int64
    "Correlation from rejuvented continuous parameter"
    ρ::Vector{T}
    "Normalized Log weights"
    ℓweightsₙ::Vector{Float64}
    "ESS and accepted steps for SMC kernel"
    ESS::Float64
    "Boolean if step has been resampled."
    resampled::Bool
    "Current iteration count."
    iter::Int64
    function SMCDiagnostics(
        ℓℒ::Vector{Float64},
        temperature::T,
        prediction::Vector{P},
        jitterdiagnostics::Vector{J},
        jittersteps::Int64,
        ρ::Vector{T},
        ℓweightsₙ::Vector{Float64},
        ESS::Float64,
        resampled::Bool,
        iter::Int64,
    ) where {P,J,T<:AbstractFloat}
        return new{P,J,T}(
            ℓℒ,
            temperature,
            prediction,
            jitterdiagnostics,
            jittersteps,
            ρ,
            ℓweightsₙ,
            ESS,
            resampled,
            iter,
        )
    end
end

############################################################################################
function generate_showvalues(diagnostics::D) where {D<:SMCDiagnostics}
    return function showvalues()
        return (:smc, "diagnostics"),
        (:iter, diagnostics.iter),
        (:AvgLogLik, mean(diagnostics.ℓℒ)),
        (:ESS, diagnostics.ESS),
        (:resampled, diagnostics.resampled),
        (:AvgJitterCorrelation, mean(diagnostics.ρ)),
        (:Temperature, diagnostics.temperature)
    end
end

############################################################################################
#export
export SMCDiagnostics, generate_showvalues
