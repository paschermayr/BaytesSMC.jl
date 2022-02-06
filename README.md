# BaytesSMC

<!---
![logo](docs/src/assets/logo.svg)
[![CI](xxx)](xxx)
[![arXiv article](xxx)](xxx)
-->

[![Documentation, Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://paschermayr.github.io/BaytesSMC.jl/)
[![Build Status](https://github.com/paschermayr/BaytesSMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/paschermayr/BaytesSMC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/paschermayr/BaytesSMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/paschermayr/BaytesSMC.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

BaytesSMC.jl is a library to perform SMC proposal steps on `ModelWrapper` structs, see [ModelWrappers.jl](https://github.com/paschermayr/ModelWrappers.jl). Kernels that are defined in [BaytesMCMC.jl](https://github.com/paschermayr/BaytesMCMC.jl) and [BaytesFilters.jl](https://github.com/paschermayr/BaytesFilters.jl) can be
used inside this library.

BaytesSMC.jl supports sequential parameter estimation frameworks such as IBIS and SMC2. This can be achieved via data tempering, which is explained in more detail in the upcoming Baytes.jl library. Moreover, a SMC variant for batch data is provided as well, where tempering of the objective function is used until the temperature reaches 1.0.


<!---
[BaytesMCMC.jl](xxx)
[BaytesFilters.jl](xxx)
[BaytesPMCMC.jl](xxx)
[BaytesSMC.jl](xxx)
[Baytes.jl](xxx)
-->

## First steps

All standard Baytes.jl functions call can be used in BaytesSMC.jl. To start with, we have
to define parameter and an objective function first.
Let us use the model initially defined in the [ModelWrappers.jl](https://github.com/paschermayr/ModelWrappers.jl) introduction:
```julia
using ModelWrappers, BaytesSMC
using Distributions, Random, UnPack
_rng = Random.GLOBAL_RNG
#Create Model and data
myparameter = (μ = Param(0.0, Normal()), σ = Param(1.0, Gamma()))
mymodel = ModelWrapper(myparameter)
data = randn(1000)
#Create objective for both μ and σ and define a target function for it
myobjective = Objective(mymodel, data, (:μ, :σ))
function (objective::Objective{<:ModelWrapper{BaseModel}})(θ::NamedTuple)
	@unpack data = objective
	lprior = Distributions.logpdf(Distributions.Normal(),θ.μ) + Distributions.logpdf(Distributions.Exponential(), θ.σ)
    llik = sum(Distributions.logpdf( Distributions.Normal(θ.μ, θ.σ), data[iter] ) for iter in eachindex(data))
	return lprior + llik
end
```

A particle in the SMC framework corresponds to a full model. We will assign assign a NUTS sampler for each particle, and propose new parameter via SMC in the standard Bayes.jl framework:
```julia
using BaytesMCMC
smc = SMC(_rng, MCMC(NUTS,(:μ, :σ,)), myobjective)
propose!(_rng, smc, mymodel, data)
```
## Customization

Construction is highly flexible, and can be tweaked via keyword assignments in the following helper structs:
```julia
kerneldefault  = SMCDefault(Ntuning = 50, jittermin = 1, jittermax = 10)
samplingdefault = SampleDefault(chains = 10)
SMC(_rng, MCMC(NUTS,(:μ, :σ,)), myobjective, kerneldefault, samplingdefault)
```
`kerneldefault` consists of tuning parameter that are specific for SMC,
`samplingdefault` consists of tuning parameter that are observed over the whole sampling process. The latter is explained in more detail in Baytes.jl.

## Going Forward

This package is still highly experimental - suggestions and comments are always welcome!

<!---
# Citing Baytes.jl

If you use Baytes.jl for your own research, please consider citing the following publication: ...
-->
