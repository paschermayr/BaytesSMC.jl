############################################################################################
# Models to be used in construction
objectives = [Objective(ModelWrapper(MyBaseModel(), myparameter, FlattenDefault()), data_uv),
    Objective(ModelWrapper(MyBaseModel(), myparameter, FlattenDefault(; output = Float32)), data_uv)
    ]

## Make model for several parameter types
for iter in eachindex(objectives)
    _obj = objectives[iter]
    _flattentype = _obj.model.info.flattendefault.output
    @testset "Kernel construction and propagation, all models" begin
        ## Assign smc kernel
        mcmc = MCMC(NUTS,(:μ, :σ,); config_kw = (;stepsizeadaption=UpdateFalse()))
        kerneldefault  = SMCDefault(Ntuning = 50, jittermin = 1, jittermax = 10)
        samplingdefault = SampleDefault(chains = 10)
        SMC(_rng, mcmc, _obj, kerneldefault, samplingdefault)
        ## Propose new parameter
        smc = SMC(_rng, mcmc, _obj)
        vals, diagnostics = propose!(_rng, smc, _obj.model, _obj.data)
        @test typeof(diagnostics.temperature) == _flattentype
    end
end
