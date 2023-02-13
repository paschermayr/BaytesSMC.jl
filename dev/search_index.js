var documenterSearchIndex = {"docs":
[{"location":"intro/#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"intro/","page":"Introduction","title":"Introduction","text":"Yet to be done.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = BaytesSMC","category":"page"},{"location":"#BaytesSMC","page":"Home","title":"BaytesSMC","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BaytesSMC.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [BaytesSMC]","category":"page"},{"location":"#BaytesSMC.BaytesSMC","page":"Home","title":"BaytesSMC.BaytesSMC","text":"Sequential Monte Carlo module - MCMC and PF algorithm compositionally used\n\n\n\n\n\n","category":"module"},{"location":"#BaytesSMC.CorrelationBuffer","page":"Home","title":"BaytesSMC.CorrelationBuffer","text":"struct CorrelationBuffer{T<:AbstractFloat}\n\nStores buffers for computing correlation of model parameter particles.\n\nFields\n\nρ::Vector{T} where T<:AbstractFloat\nCorrelation between old and rejuvenated parameter\npairs::Array{Vector{T}, 1} where T<:AbstractFloat\nPairs of old and new parameter samples - for the same parameter\n\n\n\n\n\n","category":"type"},{"location":"#BaytesSMC.SMC","page":"Home","title":"BaytesSMC.SMC","text":"struct SMC{A<:SMCParticles, B<:SMCTune} <: BaytesCore.AbstractAlgorithm\n\nSMC Algorithm.\n\nFields\n\nparticles::SMCParticles\ntune::SMCTune\n\n\n\n\n\n","category":"type"},{"location":"#BaytesSMC.SMC2Kernel","page":"Home","title":"BaytesSMC.SMC2Kernel","text":"struct SMC2Kernel{P<:BaytesFilters.ParticleFilter, M<:BaytesPMCMC.PMCMC} <: BaytesCore.AbstractAlgorithm\n\nSMC2 Kernel, consisting of a kernel for the particle trajectory and kernel for all model parameter.\n\nFields\n\npf::BaytesFilters.ParticleFilter\nKernel for (online) state propagation over time.\npmcmc::BaytesPMCMC.PMCMC\nKernel for jittering step.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesSMC.SMCBuffer","page":"Home","title":"BaytesSMC.SMCBuffer","text":"struct SMCBuffer{M<:NamedTuple, I<:Integer, L<:Real, T<:AbstractFloat, F<:AbstractFloat, P, J}\n\nBuffer struct to inplace store intermediate values.\n\nFields\n\ncumweights::Vector{T} where T<:AbstractFloat\nBuffer for cumulative particle weights at each iteration, such that they can be used again next iteration after resampling step.\nparameter::BaytesCore.ModelParameterBuffer{M, L, T, I} where {M<:NamedTuple, I<:Integer, L<:Real, T<:AbstractFloat}\nBuffer for model parameter and reweighting statistics.\ncorrelation::BaytesSMC.CorrelationBuffer\nBuffer to compute correlation between original and jittered parameter.\npredictions::Vector\nPredictions\njitterdiagnostics::Vector\nDiagnostics of jitter steps. If not jittered this iteration (i.e., accepted == true), contains jitterdiagnostics from previous step.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesSMC.SMCConstructor","page":"Home","title":"BaytesSMC.SMCConstructor","text":"Callable struct to make initializing SMC sampler easier in sampling library.\n\nExamples\n\n\n\n\n\n\n\n","category":"type"},{"location":"#BaytesSMC.SMCDefault","page":"Home","title":"BaytesSMC.SMCDefault","text":"struct SMCDefault{F<:Function, T<:BaytesCore.ResamplingMethod, B<:UpdateBool, C<:UpdateBool, U<:UpdateBool}\n\nDefault arguments for SMC constructor.\n\nFields\n\nNtuning::Int64\nNumber of tuning steps used when constructing sampler.\nresamplingmethod::BaytesCore.ResamplingMethod\nThreshold for resampling chains.\nresamplingthreshold::Float64\nThreshold for resampling chains. Set to 1.0 if resampling should always be applied.\njitterfun::Function\nFunction that is applied to correlation vector of parameter to determine if jittering can be stopped.\njitteradaption::UpdateBool\nBoolean if fixed number of jittersteps are applied or jittering is based on parameter correlation.\njitterdiagnostics::UpdateBool\nBoolean if diagnostics in jittersteps should be recorded in SMCDiagnostics.\njitterthreshold::Float64\nStopping threshold against jitterfun(correlation) of jittered model parameter.\njittermin::Int64\nMinimum number of jittering steps.\njittermax::Int64\nMaximum number of jittering steps.\ngenerated::UpdateBool\nBoolean if generate(_rng, objective) for corresponding model is stored in PF Diagnostics.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesSMC.SMCDiagnostics","page":"Home","title":"BaytesSMC.SMCDiagnostics","text":"struct SMCDiagnostics{P, J, T<:AbstractFloat, G, A} <: BaytesCore.AbstractDiagnostics\n\nSMC Diagnostics container, including diagnostics of kernels used in jittering step.\n\nFields\n\nbase::BaytesCore.BaseDiagnostics{Vector{P}} where P\nDiagnostics used for all Baytes kernels\nℓincrement::Float64\nWeighted average of incremental log weights - can be used for marginal or log predictive likelihood computation.\nℓweights::Vector{Float64}\nCumulative log weights. If resampled, updated with jittered parameter - as weights will be used for tempering.\nℓweightsₜ::Vector{Float64}\nLog weights at current iteration, used for resampling criterion. NOT updated with jittered theta parameter as ℓweightsₜ depends on t-1 and jittered theta parameter come in at next iteration.\nℓweightsₙ::Vector{Float64}\nNormalized Log weights. They might be adjusted from previous iterations, so will differ from normalized ℓweightsₜ. Also, will be saved after possible resampling step, in which case all weights are set to log(1/N).\njitterdiagnostics::Vector\nDiagnostics of jitter steps. If not resampled this iteration (i.e., resampled == false), contains jitterdiagnostics from previous step.\njittersteps::Int64\nNumber of jittering steps\nρ::Vector{T} where T<:AbstractFloat\nCorrelation from rejuvented continuous parameter\nESS::Float64\nESS and accepted steps for SMC kernel\nresampled::Bool\nBoolean if step has been resampled.\ngenerated::Any\nGenerated quantities specified for objective\ngenerated_algorithm::Any\nGenerated quantities specified for algorithm\n\n\n\n\n\n","category":"type"},{"location":"#BaytesSMC.SMCParticles","page":"Home","title":"BaytesSMC.SMCParticles","text":"struct SMCParticles{B<:BaytesCore.AbstractAlgorithm, A<:ModelWrappers.ModelWrapper, D<:SMCBuffer}\n\nSMC kernel containing particles, dynamics and buffers for sampling.\n\nFields\n\nmodel::Vector{A} where A<:ModelWrappers.ModelWrapper\nSMC particles used for sampling.\nkernel::Vector{B} where B<:BaytesCore.AbstractAlgorithm\nIndividual kernels associated to an SMC particle.\nweights::BaytesCore.ParameterWeights\nWeights struct for all SMC particles.\nbuffer::SMCBuffer\nBuffer values for allocation free evaluations.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesSMC.SMCParticles-Tuple{Random.AbstractRNG, BaytesCore.AbstractConstructor, ModelWrappers.Objective, BaytesCore.SampleDefault, SMCTune}","page":"Home","title":"BaytesSMC.SMCParticles","text":"SMCParticles(_rng, JitterKernel, objective, info, tune)\n\n\nInitialize SMC particles.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesSMC.SMCParticles-Tuple{Random.AbstractRNG, SMC2Constructor, ModelWrappers.Objective, BaytesCore.SampleDefault, SMCTune}","page":"Home","title":"BaytesSMC.SMCParticles","text":"SMCParticles(_rng, kernel, objective, info, tune)\n\n\nInitialize SMC² sampler.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesSMC.SMCTune","page":"Home","title":"BaytesSMC.SMCTune","text":"struct SMCTune{T<:ModelWrappers.Tagged, F<:Function, B<:UpdateBool, C<:UpdateBool, R<:BaytesCore.ResamplingMethod, U<:UpdateBool, S<:UpdateBool, V<:BaytesCore.ValueHolder}\n\nSMC tuning container.\n\nFields\n\ntagged::ModelWrappers.Tagged\nTagged Model parameter.\niter::BaytesCore.Iterator\nIterator that contains current data iteration ~ starts with Ndata.\nchains::BaytesCore.ChainsTune\nParticleTune struct to possibly adaptively change number of chains.\nresample::BaytesCore.ResampleTune\nResampling Method of parameter particle indices.\njitter::BaytesCore.JitterTune\nFunction that is applied to correlation of parameter to determine if jittering can be stopped.\njitterfun::Function\nDetermines number of jittering steps in jitter kernel.\njitterdiagnostics::UpdateBool\nBoolean if diagnostics in jittersteps should be recorded in SMCDiagnostics.\ncapture::UpdateBool\nUpdateBool if parameter can be captured after initial jitter step.\nNtuning::Int64\nNumber of warmup Tuning steps for kernels.\nbatchsize::Int64\nBatchsize for parallel processing of jittering steps.\ngenerated::UpdateBool\nBoolean if generated quantities should be generated while sampling\ntemperatureₜ₋₁::BaytesCore.ValueHolder\nHolds temperature of previous iteration so data tempering can be beformed for resample step.\n\n\n\n\n\n","category":"type"},{"location":"#BaytesCore.generate-Tuple{Random.AbstractRNG, SMC, ModelWrappers.Objective}","page":"Home","title":"BaytesCore.generate","text":"generate(_rng, algorithm, objective)\n\n\nGenerate statistics for algorithm given model parameter and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.infer-Union{Tuple{D}, Tuple{Random.AbstractRNG, Type{BaytesCore.AbstractDiagnostics}, SMC, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesCore.infer","text":"infer(_rng, diagnostics, smc, model, data)\n\n\nInfer MCMC diagnostics type.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.jitter!-Union{Tuple{P}, Tuple{D}, Tuple{Random.AbstractRNG, SMCParticles, SMCTune, D, P}} where {D, P<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesCore.jitter!","text":"Jitter θ particles with given kernels. This is performed in 2 stages:\n\nkernel is updated with new data and shuffled particles, then one proposal step is performed.\nIf more than one jitterstep has to be performed, previous results might be captured depending on the kernel.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propagate!-Union{Tuple{P}, Tuple{D}, Tuple{Random.AbstractRNG, SMCParticles, SMCTune, D, P}} where {D, P<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesCore.propagate!","text":"Propagate data forward over time. If (latent) data has to be extended, need to overload this function for specific smc kernel.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propagate!-Union{Tuple{P}, Tuple{D}, Tuple{Random.AbstractRNG, SMCParticles{<:SMC2Kernel}, SMCTune, D, P}} where {D, P<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesCore.propagate!","text":"propagate!(_rng, particles, tune, data, proposaltune)\n\n\nPropagate data forward over time.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.propose!-Union{Tuple{T}, Tuple{D}, Tuple{Random.AbstractRNG, SMC, ModelWrappers.ModelWrapper, D}, Tuple{Random.AbstractRNG, SMC, ModelWrappers.ModelWrapper, D, T}} where {D, T<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesCore.propose!","text":"propose!(_rng, smc, model, data)\npropose!(_rng, smc, model, data, proposaltune)\n\n\nPropose new parameter with smc sampler. If update=true, objective function will be updated with input model and data.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.resample!-Union{Tuple{P}, Tuple{D}, Tuple{Random.AbstractRNG, SMCParticles, SMCTune, D, P}} where {D, P<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesCore.resample!","text":"Resample particle ancestors, shuffle current particles and rejuvenate θ.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.results-Union{Tuple{M}, Tuple{T}, Tuple{AbstractVector{M}, SMC, Int64, Vector{T}}} where {T<:Real, M<:SMCDiagnostics}","page":"Home","title":"BaytesCore.results","text":"results(diagnosticsᵛ, smc, Ndigits, quantiles)\n\n\nPrint diagnostics result of SMC sampler.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesCore.weight!-Union{Tuple{P}, Tuple{D}, Tuple{Random.AbstractRNG, SMCParticles, SMCTune, D, P}} where {D, P<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesCore.weight!","text":"Compute new cumulative and incremental particle weights.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesSMC.SMCreweight-Union{Tuple{P}, Tuple{Random.AbstractRNG, Any, ModelWrappers.Objective, P, Any}} where P<:BaytesCore.ProposalTune","page":"Home","title":"BaytesSMC.SMCreweight","text":"SMCreweight(_rng, algorithm, objective, proposaltune, cumweightsₜ₋₁)\n\n\nComputes particle weights after jiterring step. Defaults to SMCweight function.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesSMC.SMCweight-Union{Tuple{P}, Tuple{Random.AbstractRNG, Any, ModelWrappers.Objective, P, Any}} where P<:BaytesCore.ProposalTune","page":"Home","title":"BaytesSMC.SMCweight","text":"Compute cumulative and incremental weight of objective at time/iteration t, given weight at t-1. Incremental weight will be used as particle weight for resampling. Cumulative weight will be used to adapt temperature.\n\nIf temperature is constant, this function can be overloaded with your ModelName if incremental weight can be computed independent of previous weight, which speeds up computation. cumweightsₜ is not needed in this case.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesSMC.compute_batchsize","page":"Home","title":"BaytesSMC.compute_batchsize","text":"compute_batchsize(Nchains)\ncompute_batchsize(Nchains, Nthreads)\n\n\nCompute maximum number for batchsize in resampling step of SMC.\n\nExamples\n\n\n\n\n\n\n\n","category":"function"},{"location":"#BaytesSMC.compute_ρ!-Union{Tuple{F}, Tuple{T}, Tuple{S}, Tuple{Vector{T}, Array{Vector{F}, 1}, Array{Vector{S}, 1}, Any}, Tuple{Vector{T}, Array{Vector{F}, 1}, Array{Vector{S}, 1}, Any, Any}} where {S<:Real, T<:AbstractFloat, F<:Real}","page":"Home","title":"BaytesSMC.compute_ρ!","text":"compute_ρ!(ρ, pairs, valᵤ, algorithmᵛ)\ncompute_ρ!(ρ, pairs, valᵤ, algorithmᵛ, ϵ)\n\n\nCompute inplace the correlation between old parameter vectors valᵤ and new parameter vectors from 'algorithmᵛ'.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesSMC.infer_generated-Union{Tuple{D}, Tuple{Random.AbstractRNG, SMC, ModelWrappers.ModelWrapper, D}} where D","page":"Home","title":"BaytesSMC.infer_generated","text":"infer_generated(_rng, smc, model, data)\n\n\nInfer type of generated quantities of PF sampler.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesSMC.jitterkernel-Tuple{SMCParticles, Int64}","page":"Home","title":"BaytesSMC.jitterkernel","text":"jitterkernel(particles, iter)\n\n\nReturn kernel that is used for jittering.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesSMC.predict!-Union{Tuple{P}, Tuple{D}, Tuple{Random.AbstractRNG, SMCParticles, SMCTune, D, P}} where {D, P<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesSMC.predict!","text":"Predict new data for each smc particle.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"},{"location":"#BaytesSMC.reweight!-Union{Tuple{P}, Tuple{D}, Tuple{Random.AbstractRNG, SMCParticles, SMCTune, D, P}} where {D, P<:BaytesCore.ProposalTune}","page":"Home","title":"BaytesSMC.reweight!","text":"reweight!(_rng, particles, tune, data, proposaltune)\n\n\nCompute new cumulative particle weights, accounting for jittering steps. This is only needed for next iteration's weight calculation, and will not adjust current incremental and normalized weight.\n\nExamples\n\n\n\n\n\n\n\n","category":"method"}]
}
