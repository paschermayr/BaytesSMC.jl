############################################################################################
# Models to be used in construction
objectives = [
    Objective(ModelWrapper(MyBaseModel(), myparameter1, (;), FlattenDefault()), data_uv),
    Objective(ModelWrapper(MyBaseModel(), myparameter1, (;), FlattenDefault(; output = Float32)), data_uv)
]
generated = [UpdateFalse(), UpdateTrue()]

#=
iter=2
=#
############################################################################################
## Make model for several parameter types
for iter in eachindex(objectives)
    _obj = objectives[iter]
    _flattentype = _obj.model.info.reconstruct.default.output
    @testset "Kernel construction and propagation, ASMC with MCMC" begin
        ## MCMC
        mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
        kerneldefault  = SMCDefault(Ntuning = 5, jittermin = 1, jittermax = 5)
        samplingdefault = SampleDefault(chains = 4)
        smc_c = SMCConstructor(mcmc, SMCDefault(;generated = generated[iter]))
        smc_c(_rng, _obj.model, _obj.data, BaytesCore.ProposalTune(_obj.temperature), samplingdefault)
        SMC(_rng, mcmc, _obj, kerneldefault, samplingdefault)
        @test BaytesCore.get_sym(mcmc) == BaytesSMC.get_sym(smc_c)
        ## Propose new parameter
        smc = SMC(_rng, mcmc, _obj, SMCDefault(;generated = generated[iter]))
        vals, diagnostics = propose!(_rng, smc, _obj.model, _obj.data)
        @test eltype(diagnostics.ρ) == _flattentype
        proposaltune_updatefalse = BaytesCore.ProposalTune(_obj.temperature, BaytesCore.UpdateFalse(), BaytesCore.DataTune(BaytesCore.Batch(), nothing, nothing) )
        vals, diagnostics = propose!(_rng, smc, _obj.model, _obj.data, proposaltune_updatefalse)
        @test eltype(diagnostics.ρ) == _flattentype
        ## Postprocessing
        generate_showvalues(diagnostics)()
        diagtype = infer(_rng, AbstractDiagnostics, smc, _obj.model, _obj.data)
        @test diagnostics isa diagtype
        TGenerated, TGeneratedAlgorithm = BaytesSMC.infer_generated(_rng, smc, _obj.model, _obj.data)
        @test diagnostics.generated isa TGenerated
        @test diagnostics.generated_algorithm isa TGeneratedAlgorithm
        diags = Vector{diagtype}(undef, 100)
        for iter in eachindex(diags)
            _, diags[iter] = propose!(_rng, smc, _obj.model, _obj.data)
        end
        results(diags, smc, 2, [.1, .2, .5, .8, .9])

        ## Check if Jitterdiagnostics in kernel
        smc = SMC(_rng, mcmc, _obj, SMCDefault(;jitterdiagnostics = UpdateTrue(), generated = generated[iter]))
        vals, diagnostics = propose!(_rng, smc, _obj.model, _obj.data)
        @test !isa(diagnostics.jitterdiagnostics, Vector{Nothing})
    end
end


@testset "Kernel construction and propagation, ASMC with PMCMC" begin
    ## MCMC
    pfdefault_pmcmc = ParticleFilterDefault(referencing = Ancestral(),)
    mcmcdefault = MCMCDefault(; stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
    pmcmc = ParticleGibbs(
        ParticleFilterConstructor(keys(myobjective_pf.tagged.parameter), pfdefault_pmcmc),
        MCMCConstructor(NUTS, keys(myobjective_mcmc.tagged.parameter), mcmcdefault)
    )
    ## Propose new parameter
    @test length(myobjective_mcmc.model.val.latent) == length(myobjective_mcmc.data)
    smc = SMC(_rng, pmcmc, myobjective_mcmc)
    proposaltune_updatetrue = BaytesCore.ProposalTune(myobjective_mcmc.temperature, BaytesCore.UpdateTrue(), BaytesCore.DataTune(BaytesCore.Batch(), nothing, nothing) )
    vals, diagnostics = propose!(
        _rng,
        smc,
        myobjective_mcmc.model,
        myobjective_mcmc.data,
        proposaltune_updatetrue
    )

    ## Postprocessing
    diagtype = infer(_rng, AbstractDiagnostics, smc, myobjective_mcmc.model, myobjective_mcmc.data)
    @test diagnostics isa diagtype
    smc = SMC(_rng, pmcmc, myobjective_mcmc)
    diags = Vector{diagtype}(undef, 100)
    for iter in eachindex(diags)
        _, diags[iter] = propose!(
            _rng,
            smc,
            myobjective_mcmc.model,
            myobjective_mcmc.data,
            proposaltune_updatetrue
        )
    end
    #sum( diags[iter].resampled for iter in eachindex(diags) )
    results(diags, smc, 2, [.1, .2, .5, .8, .9])
end

############################################################################################
## Check IBIS
for iter in eachindex(objectives)
    _obj = objectives[iter]
    _flattentype = _obj.model.info.reconstruct.default.output
    @testset "Kernel construction and propagation, IBIS" begin
        ## Assign smc kernel ~ use PMCMC to check both MCMC and Particle Filter
        ## MCMC
        mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
        kerneldefault  = SMCDefault(Ntuning = 5, jittermin = 1, jittermax = 5)
        samplingdefault = SampleDefault(chains = 4)
        smc_c = SMCConstructor(mcmc, SMCDefault())

        #Create new objective with small amount of data
        _obj2 = Objective(_obj.model, _obj.data[1:50])
        SMC(_rng, mcmc, _obj2, kerneldefault, samplingdefault)
        @test BaytesCore.get_sym(mcmc) == BaytesSMC.get_sym(smc_c)
        ## Propose new parameter
        smc = SMC(_rng, mcmc, _obj2)
        proposaltune_updatetrue = BaytesCore.ProposalTune(_obj2.temperature, BaytesCore.UpdateTrue(), BaytesCore.DataTune(_obj2.data, BaytesCore.Expanding( length( _obj2.data ) ) ) )
        vals, diagnostics = propose!(_rng, smc, _obj2.model, _obj2.data, proposaltune_updatetrue)
        @test eltype(diagnostics.ρ) == _flattentype
        ## Postprocessing
        smc = SMC(_rng, mcmc, _obj2)
        diagtype = infer(_rng, AbstractDiagnostics, smc, _obj.model, _obj.data)
        @test diagnostics isa diagtype
        datadiff = length(_obj.data) - length(_obj2.data) - 1

        diags = Vector{diagtype}(undef, datadiff)
        for iter in eachindex(diags)
            data_temp = _obj.data[1:(length(_obj2.data) + iter)]
            proposaltune_updatetrue = BaytesCore.ProposalTune(_obj.temperature, BaytesCore.UpdateTrue(), BaytesCore.DataTune(data_temp, BaytesCore.Expanding( length(data_temp) ) ) )
            _, diags[iter] = propose!(_rng, smc, _obj.model, data_temp, proposaltune_updatetrue)
        end
        results(diags, smc, 2, [.1, .2, .5, .8, .9])
    end
end

############################################################################################
## Check SMC2
@testset "Kernel construction and propagation, SMC2" begin
    ## MCMC
    pfdefault_propagate = ParticleFilterDefault(referencing = Marginal(),)
    pfdefault_pmcmc = ParticleFilterDefault(referencing = Ancestral(),)
    mcmcdefault = MCMCDefault(; stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))

    pmcmc = ParticleGibbs(
        ParticleFilterConstructor(keys(myobjective_pf.tagged.parameter), pfdefault_pmcmc),
        MCMCConstructor(NUTS, keys(myobjective_mcmc.tagged.parameter), mcmcdefault)
    )
    pf = ParticleFilterConstructor(keys(myobjective_pf.tagged.parameter), pfdefault_propagate)
    smc2 = SMC2Constructor(pf, pmcmc)
    SMC2(pf, pmcmc)
    @test keys(myobjective_mcmc.model.val) == BaytesSMC.get_sym(smc2)

    ## Propose new parameter and exten tagged propagation parameter
    _obj = Objective(deepcopy(mymodel), data_init)
    @test length(_obj.model.val.latent) == length(_obj.data)  == N_SMC2
    smc = SMC(_rng, smc2, _obj)

    proposaltune_updatetrue = BaytesCore.ProposalTune(_obj.temperature, BaytesCore.UpdateTrue(), BaytesCore.DataTune(data[1:N_SMC2+1], BaytesCore.Expanding( length(data[1:N_SMC2+1]) ) ) )
    vals, diagnostics = propose!(
        _rng,
        smc,
        _obj.model,
        data[1:N_SMC2+1],
        proposaltune_updatetrue
    )
    @test length(mymodel.val.latent) == N_SMC2
    @test length(_obj.model.val.latent) == N_SMC2 + 1
    ## Postprocessing
    diagtype = infer(_rng, AbstractDiagnostics, smc, _obj.model, _obj.data)
    @test diagnostics isa diagtype

    _obj = Objective(deepcopy(mymodel), data_init)
    smc = SMC(_rng, smc2, _obj)
    diags = Vector{diagtype}(undef, 100)
    for iter in eachindex(diags)
        data_temp = data[1:N_SMC2+iter]
        proposaltune_updatetrue = BaytesCore.ProposalTune(_obj.temperature, BaytesCore.UpdateTrue(), BaytesCore.DataTune(data_temp, BaytesCore.Expanding( length(data_temp) ) ) )
        _, diags[iter] = propose!(
            _rng,
            smc,
            _obj.model,
            data_temp,
            proposaltune_updatetrue
        )
    end
    results(diags, smc, 2, [.1, .2, .5, .8, .9])

    ## Check if Jitterdiagnostics in kernel
    _obj = Objective(deepcopy(mymodel), data_init)
    smc = SMC(_rng, smc2, _obj, SMCDefault(;jitterdiagnostics = UpdateTrue(),))
    proposaltune_updatetrue = BaytesCore.ProposalTune(_obj.temperature, BaytesCore.UpdateTrue(), BaytesCore.DataTune(data[1:length(data_init)+1], BaytesCore.Expanding( length(data[1:length(data_init)+1]) ) ) )
    vals, diagnostics = propose!(_rng, smc, _obj.model, data[1:length(data_init)+1], proposaltune_updatetrue)
    @test !isa(diagnostics.jitterdiagnostics, Vector{Nothing})
end
############################################################################################


#=
############################################################################################
# Separate testing
_rng
smc = SMC(_rng, smc2, _obj)
proposaltune = BaytesCore.ProposalTune(_obj.temperature, BaytesCore.UpdateTrue(), BaytesCore.DataTune(data[1:N_SMC2+1], BaytesCore.Expanding( length(data[1:N_SMC2+1]) ) ) )
model = deepcopy(_obj.model)
data = [rand(_rng, Normal(μ[iter], σ[iter])) for iter in latent]
data = data[1:N_SMC2+1]
smc.tune.temperatureₜ₋₁.current = 0.1

smc.particles.model
## Update kernel parameter values with non-tagged parameter from other sampler
    #!NOTE: proposaltune.update used, not proposaltuneₜ/smc.tune.captured
update!(smc.particles, model, smc.tune.tagged, proposaltune.update)
smc.particles.model
## Set back tune.jitter and update buffer to store correct information for diagnostics
smc.tune.iter.current
smc.tune.jitter.Nsteps.current
smc.particles.buffer.correlation.ρ
update!(smc.tune)
update!(smc.particles.buffer)

smc.tune.iter.current
smc.tune.jitter.Nsteps.current
smc.particles.buffer.correlation.ρ

## Resample θ with proposal tune and data from previous iteration
smc.tune.temperatureₜ₋₁.current
proposaltune.temperature
proposaltuneₜ₋₁ = BaytesCore.ProposalTune(smc.tune.temperatureₜ₋₁.current, smc.tune.capture, proposaltune.datatune)
#!NOTE: conversion will not allocate in Baytes.jl as data will be a view already, but this ensures that this works if separately called.
dataₜ₋₁ = convert(typeof(data), BaytesCore.adjust_previous(proposaltune.datatune, data))

ESS, resampled = resample!(_rng, smc.particles, smc.tune, dataₜ₋₁, proposaltuneₜ₋₁)

#=
_rng
particles = smc.particles
tune = smc.tune
data = dataₜ₋₁
proposaltune = proposaltuneₜ₋₁

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
    particles.weights.buffer
    ## Equal weight normalized weights for next iteration memory
    Base.fill!(particles.weights.ℓweightsₙ, log(1.0 / tune.chains.Nchains))
    ## Reshuffle θ and rejuvenation particles - can be done inplace via buffer
    #!NOTE: Also set cumulative particle weights smc.buffer.weights back to correct position
    BaytesCore.shuffle!(particles.buffer.parameter, particles.model, particles.kernel, particles.buffer.cumweights) #particles.kernel,
    particles.model

    ## Rejuvenate Particles
    jitter!(_rng, particles, tune, data, proposaltune)
    particles.model
    particles.model[1].val.latent
    particles.model[2].val.latent
    ## Reweight ℓweights used for tempering so have correct index and temperature for next iteration
    particles.weights
    reweight!(_rng, particles, tune, data, proposaltune)
    particles.buffer.cumweights
end
=#

## Update temperature and proposaltune to current iteration
particles = smc.particles
tune = smc.tune
tune.temperatureₜ₋₁
BaytesCore.update!(tune.temperatureₜ₋₁, proposaltune.temperature)
    #!NOTE: smc.tune.capture might differ from proposaltune.update -> in SMC jitterng, first step always with UpdateTrue, further steps may use UpdateFalse if permitted. Hence, separate proposaltuneₜ will be used.
proposaltuneₜ = BaytesCore.ProposalTune(proposaltune.temperature, tune.capture, proposaltune.datatune)
tune.temperatureₜ₋₁.current
proposaltune.temperature
proposaltuneₜ.temperature

data = [rand(_rng, Normal(μ[iter], σ[iter])) for iter in latent]
data = data[1:(length(dataₜ₋₁)+1)]
## If latent θ trajectory increasing - propagate forward
particles.model[1].val.latent
propagate!(_rng, particles, tune, data, proposaltuneₜ)
particles.model[1].val.latent
## Predict new data given current particles
BaytesSMC.predict!(_rng, particles, tune, data, proposaltuneₜ)
## Adjust weights with log likelihood INCREMENT at time t
particles.weights
BaytesSMC.weight!(_rng, particles, tune, data, proposaltuneₜ)
particles.weights

## Compute weighted average of incremental log weights - can be used for marginal or log predictive likelihood computation.
ℓincrement = BaytesCore.weightedincrement(smc.particles.weights)

=#
