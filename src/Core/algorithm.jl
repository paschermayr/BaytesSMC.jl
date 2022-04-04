############################################################################################
"""
$(TYPEDEF)

Default arguments for SMC constructor.

# Fields
$(TYPEDFIELDS)
"""
struct SMCDefault{
    F<:Function,
    T<:BaytesCore.ResamplingMethod,
    B<:BaytesCore.UpdateBool,
    U<:BaytesCore.UpdateBool
}
    "Number of tuning steps used when constructing sampler."
    Ntuning::Int64
    "Threshold for resampling chains."
    resamplingmethod::T
    "Threshold for resampling chains. Set to 1.0 if resampling should always be applied."
    resamplingthreshold::Float64
    "Function that is applied to correlation vector of parameter to determine if jittering can be stopped."
    jitterfun::F
    "Boolean if fixed number of jittersteps are applied or jittering is based on parameter correlation."
    jitteradaption::B
    "Stopping threshold against mean correlation of jittered model parameter."
    jitterthreshold::Float64
    "Minimum number of jittering steps."
    jittermin::Int64
    "Maximum number of jittering steps."
    jittermax::Int64
    "Boolean if generate(_rng, objective) for corresponding model is stored in PF Diagnostics."
    generated::U
    function SMCDefault(;
        Ntuning=10,
        resamplingmethod::T = BaytesFilters.Systematic(),
        resamplingthreshold=0.75,
        jitterfun::F = maximum,
        jitteradaption::B = BaytesCore.UpdateTrue(),
        jitterthreshold=0.9,
        jittermin=1,
        jittermax=10,
        generated=BaytesCore.UpdateFalse(),
    ) where {
    F<:Function,
    T<:BaytesCore.ResamplingMethod,
    B<:BaytesCore.UpdateBool
    }
        ArgCheck.@argcheck 0 < Ntuning
        ArgCheck.@argcheck 0.0 <= resamplingthreshold <= 1.0
        ArgCheck.@argcheck 0.0 <= jitterthreshold <= 1.0
        ArgCheck.@argcheck 0 < jittermin <= jittermax
        return new{F, T, B, typeof(generated)}(
            Ntuning,
            resamplingmethod,
            resamplingthreshold,
            jitterfun,
            jitteradaption,
            jitterthreshold,
            jittermin,
            jittermax,
            generated
            )
    end
end

################################################################################
function update!(particles::SMCParticles, model::ModelWrapper, tagged::Tagged, upd::BaytesCore.UpdateFalse)
    return nothing
end
function update!(particles::SMCParticles, model::ModelWrapper, tagged::Tagged, upd::BaytesCore.UpdateTrue)
    # Update parameter particles that are not tagged
    for iter in eachindex(particles.model)
        ModelWrappers.fill!(particles.model[iter],
            merge(model.val, ModelWrappers.subset(particles.model[iter].val, tagged.parameter))
        )
    end
    return nothing
end

############################################################################################
"""
$(TYPEDEF)

SMC Algorithm.

# Fields
$(TYPEDFIELDS)
"""
struct SMC{A<:SMCParticles,B<:SMCTune} <: AbstractAlgorithm
    particles::A
    tune::B
    function SMC(particles::A, tune::B) where {A<:SMCParticles,B<:SMCTune}
        return new{A,B}(particles, tune)
    end
end

function SMC(
    _rng::Random.AbstractRNG,
    Jitterkernel::J,
    objective::Objective,
    default::SMCDefault=SMCDefault(),
    info::BaytesCore.SampleDefault = BaytesCore.SampleDefault()
) where {
    J<:AbstractConstructor
}
    ## Print assumptions to user
    @unpack chains = info
    @unpack Ntuning, resamplingmethod, resamplingthreshold, jitterfun, jitteradaption, jitterthreshold, jittermin, jittermax, generated = default
    Ndata = maximum(size(objective.data))
    #!NOTE: Check if steps after first jitter call can be captured
    capture = Jitterkernel isa MCMCConstructor ? BaytesCore.UpdateFalse() : BaytesCore.UpdateTrue()
    ## Initialize tune
    tagged = ModelWrappers.Tagged(objective.model, BaytesCore.get_sym(Jitterkernel))
    tune = SMCTune(
        tagged,
        chains,
        Ndata,
        Ntuning,
        resamplingthreshold,
        resamplingmethod,
        capture,
        jitterfun,
        jitteradaption,
        jitterthreshold,
        jittermin,
        jittermax,
        generated
    )
    ## Initialize SMCParticles
    particles = SMCParticles(_rng, Jitterkernel, objective, info, tune)
    ## Return SMC container
    return SMC(particles, tune)
end

############################################################################################
"""
$(SIGNATURES)
Propose new parameter with smc sampler. If update=true, objective function will be updated with input model and data.

# Examples
```julia
```

"""
function propose!(
    _rng::Random.AbstractRNG,
    smc::SMC,
    model::ModelWrapper,
    data::D,
    temperature::F = model.info.flattendefault.output(1.0),
    update::U=BaytesCore.UpdateTrue()
) where {D,F<:AbstractFloat,U<:BaytesCore.UpdateBool}
    ## Update kernel parameter values with non-tagged parameter from other sampler
    update!(smc.particles, model, smc.tune.tagged, update)
    ## Set back tune.jitter and update buffer to store correct information for diagnostics
    update!(smc.tune)
    update!(smc.particles.buffer)
    ## If latent θ trajectory increasing - propagate forward
    propagate!(_rng, smc.particles, smc.tune, data, temperature)
    ## Predict new data given current particles
    predict!(_rng, smc.particles, smc.tune, data, temperature)
    ## Adjust weights with log likelihood INCREMENT at time t
    weight!(_rng, smc.particles, smc.tune, data, temperature)
    ## Resample and jitter particles if criterion fulfilled
    #!NOTE: Cannot use resample before propagate! (as in pf) as previous temperature otherwise unknown
    ESS, resampled = resample!(_rng, smc.particles, smc.tune, data, temperature)
    ## Choose proposal model parameter to update model.val
    path = BaytesFilters.draw!(_rng, smc.particles.weights)
    ModelWrappers.fill!(model,
        merge(model.val, ModelWrappers.subset(smc.particles.model[path].val, smc.tune.tagged.parameter))
    )
    ## Pack and return Models and particles
    return model.val,
    SMCDiagnostics(
        smc.particles,
        temperature,
        ESS,
        resampled,
        smc.tune.jitter.Nsteps.current,
        smc.tune.iter.current,
        ModelWrappers.generate(_rng, Objective(model, data, smc.tune.tagged, temperature), smc.tune.generated)
    )
end

############################################################################################
#export
export SMC, SMCDefault, propose!
