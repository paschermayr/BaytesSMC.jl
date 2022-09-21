############################################################################################
"""
$(TYPEDEF)

SMC tuning container.

# Fields
$(TYPEDFIELDS)
"""
struct SMCTune{
    T<:Tagged,
    F<:Function,
    B<:BaytesCore.UpdateBool,
    C<:BaytesCore.UpdateBool,
    R<:BaytesCore.ResamplingMethod,
    U<:BaytesCore.UpdateBool,
    S<:BaytesCore.UpdateBool,
    V<:BaytesCore.ValueHolder
    }
    "Tagged Model parameter."
    tagged::T
    "Iterator that contains current data iteration ~ starts with Ndata."
    iter::Iterator
    "ParticleTune struct to possibly adaptively change number of chains."
    chains::BaytesCore.ChainsTune
    "Resampling Method of parameter particle indices."
    resample::BaytesCore.ResampleTune{R}
    "Function that is applied to correlation of parameter to determine if jittering can be stopped."
    jitter::BaytesCore.JitterTune{B}
    "Determines number of jittering steps in jitter kernel."
    jitterfun::F
    "Boolean if diagnostics in jittersteps should be recorded in SMCDiagnostics."
    jitterdiagnostics::C
    "UpdateBool if parameter can be captured after initial jitter step."
    capture::U
    "Number of warmup Tuning steps for kernels."
    Ntuning::Int64
    "Batchsize for parallel processing of jittering steps."
    batchsize::Int64
    "Boolean if generated quantities should be generated while sampling"
    generated::S
    "Holds temperature of previous iteration so data tempering can be beformed for resample step."
    temperatureₜ₋₁::V
    function SMCTune(
        tagged::T,
        Nchains::Int64,
        Ndata::Int64,
        Ntuning::Int64,
        resamplingthreshold::Float64,
        resampling::D,
        capture::U,
        jitterfun::F,
        jitteradaption::B,
        jitterdiagnostics::C,
        jitterthreshold::Float64,
        Njitter_min::Integer,
        Njitter_max::Integer,
        generated::S,
        temperature::A
    ) where {
        T<:Tagged,
        F<:Function,
        B<:BaytesCore.UpdateBool,
        C<:BaytesCore.UpdateBool,
        D<:BaytesCore.ResamplingMethod,
        U<:BaytesCore.UpdateBool,
        S<:BaytesCore.UpdateBool,
        A<:Real
    }
        ## Check input for correctness
        ArgCheck.@argcheck 0 < Ntuning "Number of tuning iterations has to be positive"
        ArgCheck.@argcheck 0.0 <= resamplingthreshold "threshold has to be positive"
        ArgCheck.@argcheck 0.0 <= jitterthreshold "jitterthreshold has to be positive"
        ArgCheck.@argcheck 0 < Njitter_min <= Njitter_max
        ArgCheck.@argcheck 0.0 <= temperature <= 1.0
        ## Assign tuning structs
        jitter_maybe = Updater(false)
        #!NOTE: Start with Ndata, so first propose!() step has correct iteration after training.
        iter = Iterator(Ndata)
        particletune = BaytesCore.ChainsTune(
            Nchains / Ndata, resamplingthreshold, Nchains, Ndata
        )
        jitter = BaytesCore.JitterTune(jitteradaption, jitterthreshold, Njitter_min, Njitter_max)
        resample = BaytesCore.ResampleTune(resampling, BaytesCore.Updater(false))
        ## Compute batchsize for parallel processing in resampling step
        batchsize = compute_batchsize(Nchains)
        ## Save current temperature
        temperatureₜ₋₁ = BaytesCore.ValueHolder(temperature)
        ## Return SMC tune
        return new{T,F,B,C,D,U,S, typeof(temperatureₜ₋₁)}(
            tagged, iter, particletune, resample, jitter, jitterfun, jitterdiagnostics, capture, Ntuning, batchsize, generated, temperatureₜ₋₁
        )
    end
end

############################################################################################
function update!(tune::SMCTune)
    ##  Update iteration counter
    update!(tune.iter)
    ## Set jitter count back to 0
    update!(tune.jitter)
    return nothing
end

############################################################################################
#export
export SMCTune
