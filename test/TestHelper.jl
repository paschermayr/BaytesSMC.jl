############################################################################################
# Constants
"RNG for sampling based solutions"
const _rng = Random.MersenneTwister(1)   # shorthand
Random.seed!(_rng, 1)

"Tolerance for stochastic solutions"
const _TOL = 1.0e-6

"Number of samples"
N = 10^3
N_SMC2 = 10^2
############################################################################################
# Kernel and AD backends

############################################################################################
# Initiate Base Model to check sampler

######################################## Model 1
struct MyBaseModel <: ModelName end
myparameter1 = (μ = Param(0.0, Normal()), σ = Param(10.0, Gamma()))
mymodel1 = ModelWrapper(MyBaseModel(), myparameter1)

data_uv = rand(_rng, Normal(mymodel1.val.μ, mymodel1.val.σ), N)
#Create objective for both μ and σ and define a target function for it
myobjective1 = Objective(mymodel1, data_uv, (:μ, :σ))
function (objective::Objective{<:ModelWrapper{MyBaseModel}})(θ::NamedTuple)
	@unpack data = objective
	lprior = Distributions.logpdf(Distributions.Normal(),θ.μ) + Distributions.logpdf(Distributions.Exponential(), θ.σ)
    llik = sum(Distributions.logpdf( Distributions.Normal(θ.μ, θ.σ), data[iter] ) for iter in eachindex(data))
	return lprior + llik
end
myobjective1(myobjective1.model.val)

function ModelWrappers.generate(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{MyBaseModel}})
    @unpack model, data = objective
    @unpack μ, σ = model.val
    return Float16(μ[1])
end

function ModelWrappers.predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{MyBaseModel}})
    @unpack model, data = objective
    @unpack μ, σ = model.val
	return rand(_rng, Normal(μ, σ))
end

######################################## Model 2 ~ PMCMC/SMC2

# Parameter
μ = [-2., 2.]
σ = [1., 1.]
p = [.05, .95]
# Latent data
latent = rand(_rng, Categorical(p), N)
data = [rand(_rng, Normal(μ[iter], σ[iter])) for iter in latent]

# Create ModelWrapper struct, assuming we do not know latent
latent_init = rand(_rng, Categorical(p), N_SMC2)
data_init = data[1:N_SMC2]

myparameter = (;
    μ = Param(μ, [Normal(-2., 5), Normal(2., 5)]),
    σ = Param(σ, [Gamma(2.,2.), Gamma(2.,2.)]),
    p = Param(p, Dirichlet(2, 2)),
    latent = Param(latent_init, [Categorical(p) for _ in Base.OneTo(N_SMC2)]),
)
mymodel = ModelWrapper(myparameter)
myobjective = Objective(mymodel, data_init)

# Assign an objective for both a particle filter and an mcmc kernel:
myobjective_pf = Objective(mymodel, data_init, :latent)
myobjective_mcmc = Objective(mymodel, data_init, (:μ, :σ, :p))

@test length(myobjective_mcmc.model.val.latent) == N_SMC2

# Assign Model dynamics
function BaytesFilters.dynamics(objective::Objective{<:ModelWrapper{BaseModel}})
    @unpack model, data = objective
    @unpack μ, σ, p = model.val

    initial_latent = Categorical(p)
    transition_latent(particles, iter) = initial_latent
    transition_data(particles, iter) = Normal(μ[particles[iter]], σ[particles[iter]])

    return Markov(initial_latent, transition_latent, transition_data)
end
dynamics(myobjective)

# Assign log target
function (objective::Objective{<:ModelWrapper{BaseModel}})(θ::NamedTuple)
    @unpack model, data, tagged = objective
    @unpack μ, σ, p, latent = θ
## Prior -> a faster shortcut without initializing the priors again
    lprior = log_prior(tagged.info.constraint, ModelWrappers.subset(θ, tagged.parameter) )
##Likelihood
    dynamicsᵉ = [Normal(μ[iter], σ[iter]) for iter in eachindex(μ)]
    dynamicsˢ = Categorical(p)
    ll = 0.0
#FOR PMCMC ~ target p(θ ∣ latent_1:t, data_1:t)
    for iter in eachindex(data)
        ll += logpdf(dynamicsᵉ[latent[iter]], data[iter])
        ll += logpdf(dynamicsˢ, latent[iter] )
    end
#=
# FOR MCMC ~ target p(θ ∣ data_1:t) by integrating out latent_1:t
    for time in eachindex(data)
        ll += logsumexp(logpdf(dynamicsˢ, iter) + logpdf(dynamicsᵉ[iter], grab(data, time)) for iter in eachindex(dynamicsᵉ))
    end
=#
    return ll + lprior
end
