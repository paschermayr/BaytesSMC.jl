############################################################################################
"""
$(SIGNATURES)
Return kernel that is used for jittering.

# Examples
```julia
```

"""
function jitterkernel(particles::SMCParticles, iter::Int64)
    return particles.kernel[iter]
end

############################################################################################
"""
$(SIGNATURES)
Resample particle ancestors, shuffle current particles and rejuvenate θ.

# Examples
```julia
```

"""
function resample!(
    _rng::Random.AbstractRNG,
    particles::SMCParticles,
    tune::SMCTune,
    data::D,
    proposaltune::P
) where {D, P<:ProposalTune}
    ## Set resample back to false from previous iteration
    init!(tune.resample, false)
    ## Compute ESS
    ESS = BaytesCore.computeESS(particles.weights)
    resampled = BaytesCore.isresampled(ESS, tune.chains.Nchains * tune.chains.threshold)
    if resampled
        ## Record in tuning for diagnostics and propagation step
        init!(tune.resample, true)
        ## Resample ancestors ~ buffer already contain normalized weights from computeESS(weights) call
        resample!(
            _rng,
            tune.resample.method,
            particles.buffer.parameter.index,
            tune.iter.current,
            particles.weights.buffer,
            tune.chains.Nchains,
        )
        ## Equal weight normalized weights for next iteration memory
        Base.fill!(particles.weights.ℓweightsₙ, log(1.0 / tune.chains.Nchains))
        ## Reshuffle θ and rejuvenation particles - can be done inplace via buffer
        #!NOTE: Also set cumulative particle weights smc.buffer.weights back to correct position
        BaytesCore.shuffle!(particles.buffer.parameter, particles.model, particles.kernel, particles.buffer.cumweights) #particles.kernel,
        ## Rejuvenate Particles
        jitter!(_rng, particles, tune, data, proposaltune)
        ## Reweight cumulative ℓweights (stored in buffer) used for tempering so have correct index and temperature for next iteration
        reweight!(_rng, particles, tune, data, proposaltune)
    end
    return ESS, resampled
end

############################################################################################
"""
$(SIGNATURES)
Jitter θ particles with given kernels. This is performed in 2 stages:
1. kernel is updated with new data and shuffled particles, then one proposal step is performed.
2. If more than one jitterstep has to be performed, previous results might be captured depending on the kernel.

# Examples
```julia
```

"""
function jitter!(_rng::Random.AbstractRNG, particles::SMCParticles, tune::SMCTune, data::D, proposaltune::P) where {D, P<:ProposalTune}
    ## Make first jitter step, update for new data and shuffled parameter
#    Polyester.@batch per=thread minbatch=tune.batchsize for iter in eachindex(particles.kernel)
    proposaltune_initial = ProposalTune(proposaltune.temperature, BaytesCore.UpdateTrue(), proposaltune.datatune)
    Base.Threads.@threads for iter in eachindex(particles.kernel)
        ## Propose new parameter
        #!NOTE: UpdateTrue() ensures that MCMC Kernel's LogDensityResult is updated with new particles.model[iter] parameter
        _, particles.buffer.jitterdiagnostics[iter] = propose!(
            _rng,
            jitterkernel(particles, iter),
            particles.model[iter],
            data,
            proposaltune_initial
        )
        #!NOTE: has to be in loop for scoping rules
        particles.buffer.predictions[iter] = BaytesCore.get_prediction(
            particles.buffer.jitterdiagnostics[iter]
        )
    end
    # Calculate Correlation between old and new θ particles
    compute_ρ!(particles.buffer.correlation, particles.buffer.parameter.valᵤ, particles.kernel)
    # Check if jittering can be stopped
    jitter = BaytesCore.jitter!(tune.jitter, tune.jitterfun(particles.buffer.correlation.ρ))
    ## If more than one jitterstep has to be performed, previous results might be captured depending on the kernel.
    #!NOTE buffer.parameter.result.θᵤ at correct place because jitter! only happens after shuffle!
    while jitter
        ## Jitter Particles
#        Polyester.@batch per=thread minbatch=tune.batchsize for iter in eachindex(particles.kernel)
        Base.Threads.@threads for iter in eachindex(particles.kernel)
            ## Propose new parameter
            #!TODO: This is only true in the first call though, if more than 1 step jittered, MCMC only kernels could be set to UpdateFalse()
            _, particles.buffer.jitterdiagnostics[iter] = propose!(
                _rng,
                jitterkernel(particles, iter),
                particles.model[iter],
                data,
                proposaltune
            )
            #!NOTE: has to be in loop for scoping rules
            particles.buffer.predictions[iter] = BaytesCore.get_prediction(
                particles.buffer.jitterdiagnostics[iter]
            )
        end
        ## Calculate Correlation between old and new θ particles
        compute_ρ!(particles.buffer.correlation, particles.buffer.parameter.valᵤ, particles.kernel)
        ## Check if jittering can be stopped
        jitter = jitter!(tune.jitter, tune.jitterfun(particles.buffer.correlation.ρ))
    end
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Propagate data forward over time. If (latent) data has to be extended, need to overload this function for specific smc kernel. I.e., SMC2 Algorithm dispatches on SMCParticles{<:SMC2Kernel}.

# Examples
```julia
```

"""
function propagate!(_rng::Random.AbstractRNG, particles::SMCParticles, tune::SMCTune, data::D, proposaltune::P) where {D, P<:ProposalTune}
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Predict new data for each smc particle.

# Examples
```julia
```

"""
function predict!(_rng::Random.AbstractRNG, particles::SMCParticles, tune::SMCTune, data::D, proposaltune::P) where {D, P<:ProposalTune}
    ## Propagate series forward with recent particle
    for iter in eachindex(particles.model)
        ## Predict new data point
        particles.buffer.predictions[iter] = predict(
            _rng,
            particles.kernel[iter],
            Objective(particles.model[iter], data, tune.tagged, proposaltune.temperature)
        )
    end
    #!NOTE: Here, particles.buffer.jitterdiagnostics will not be updated, and for SMCDiagnostics, last available jitterdiagnostics are provided.
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Compute new cumulative and incremental particle weights.

# Examples
```julia
```

"""
function weight!(
    _rng::Random.AbstractRNG,
    particles::SMCParticles,
    tune::SMCTune,
    data::D,
    proposaltune::P
) where {D, P<:ProposalTune}
    ## Compute incremental weight for resampling step, and cumulative weight for i) next iteration and ii) temperature adjustment
    @inbounds for iter in eachindex(particles.weights.ℓweights)
        objective = Objective(particles.model[iter], data, tune.tagged, proposaltune.temperature)
        #!NOTE: particles.buffer.cumweights is cumulative weight at previous iteration, accounting for jittering step.
        #!NOTE: ℓweights will be 0.0 at first iteration if not data or temperature tempering is performed
        particles.buffer.cumweights[iter], particles.weights.ℓweights[iter] = SMCweight(
            _rng,
            particles.kernel[iter],
            objective,
            proposaltune,
            particles.buffer.cumweights[iter]
        )
    end
    BaytesFilters.normalize!(particles.weights)
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Compute cumulative and incremental weight of objective at time/iteration t, given weight at t-1.
Incremental weight will be used as particle weight for resampling.
Cumulative weight will be used to adapt temperature.

If temperature is constant, this function can be overloaded with your ModelName if incremental weight can be computed independent of previous weight,
which speeds up computation. `cumweightsₜ` is not needed in this case.

# Examples
```julia
```

"""
function SMCweight(_rng::Random.AbstractRNG, algorithm, objective::Objective, proposaltune::P, cumweightsₜ₋₁) where {P<:ProposalTune}
    cumweightsₜ = objective.temperature * objective(objective.model.val)
    #!NOTE: cumweightsₜ₋₁ already has correct temperature at t-1
    return cumweightsₜ, cumweightsₜ - cumweightsₜ₋₁
end

############################################################################################
"""
$(SIGNATURES)
Compute new cumulative particle weights, accounting for jittering steps. This is only needed for
next iteration's weight calculation, and will not adjust current incremental and normalized weight.

# Examples
```julia
```

"""
function reweight!(
    _rng::Random.AbstractRNG,
    particles::SMCParticles,
    tune::SMCTune,
    data::D,
    proposaltune::P
) where {D, P<:ProposalTune}
    ## Compute incremental weight for resampling step, and cumulative weight for i) next iteration and ii) temperature adjustment
    @inbounds for iter in eachindex(particles.weights.ℓweights)
        objective = Objective(particles.model[iter], data, tune.tagged, proposaltune.temperature)
        #!NOTE: particles.buffer.cumweights is cumulative weight at previous iteration, accounting for jittering step.
        particles.buffer.cumweights[iter], _ = SMCreweight(
            _rng,
            particles.kernel[iter],
            objective,
            proposaltune,
            particles.buffer.cumweights[iter]
        )
    end
    return nothing
end

############################################################################################
"""
$(SIGNATURES)
Computes particle weights after jiterring step. Defaults to `SMCweight` function.

# Examples
```julia
```

"""
function SMCreweight(_rng::Random.AbstractRNG, algorithm, objective::Objective, proposaltune::P, cumweightsₜ₋₁) where {P<:ProposalTune}
    return SMCweight(_rng, algorithm, objective, proposaltune, cumweightsₜ₋₁)
end

############################################################################################
# Export
export
    resample!,
    jitter!,
    propagate!,
    weight!,
    reweight!,
    SMCweight,
    SMCreweight
