############################################################################################
# Models to be used in construction
objectives = [
    Objective(ModelWrapper(MyBaseModel(), myparameter, FlattenDefault()), data_uv),
    Objective(ModelWrapper(MyBaseModel(), myparameter, FlattenDefault(; output = Float32)), data_uv)
]

############################################################################################
## Make model for several parameter types
for iter in eachindex(objectives)
    _obj = objectives[iter]
    _flattentype = _obj.model.info.flattendefault.output
    @testset "Kernel construction and propagation, MCMC" begin
        ## Assign smc kernel ~ use PMCMC to check both MCMC and Particle Filter
        ## MCMC
        mcmc = MCMC(NUTS,(:μ, :σ,); stepsize = ConfigStepsize(;stepsizeadaption = UpdateFalse()))
        kerneldefault  = SMCDefault(Ntuning = 5, jittermin = 1, jittermax = 5)
        samplingdefault = SampleDefault(chains = 4)
        smc_c = SMCConstructor(mcmc, SMCDefault())
        SMC(_rng, mcmc, _obj, kerneldefault, samplingdefault)
        @test BaytesCore.get_sym(mcmc) == BaytesSMC.get_sym(smc_c)
        ## Propose new parameter
        smc = SMC(_rng, mcmc, _obj)
        vals, diagnostics = propose!(_rng, smc, _obj.model, _obj.data)
        @test eltype(diagnostics.ρ) == _flattentype
        ## Postprocessing
        diagtype = infer(_rng, AbstractDiagnostics, smc, _obj.model, _obj.data)
        @test diagnostics isa diagtype
        diags = Vector{diagtype}(undef, 100)
        for iter in eachindex(diags)
            _, diags[iter] = propose!(_rng, smc, _obj.model, _obj.data)
        end
        results(diags, smc, 2, [.1, .2, .5, .8, .9])
    end
end

############################################################################################
## Check IBIS

############################################################################################
## Check SMC2
