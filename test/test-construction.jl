############################################################################################
# Models to be used in construction
objectives = [
    Objective(ModelWrapper(MyBaseModel(), myparameter1, (;), FlattenDefault()), data_uv),
    Objective(ModelWrapper(MyBaseModel(), myparameter1, (;), FlattenDefault(; output = Float32)), data_uv)
]
generated = [UpdateFalse(), UpdateTrue()]
#=
iter=1
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
        smc_c(_rng, _obj.model, _obj.data, _obj.temperature, samplingdefault)
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
        TGenerated = BaytesSMC.infer_generated(_rng, smc, _obj.model, _obj.data)
        @test diagnostics.generated isa TGenerated
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
