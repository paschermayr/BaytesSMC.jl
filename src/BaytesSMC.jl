"Sequential Monte Carlo module - MCMC and PF algorithm compositionally used"
module BaytesSMC

############################################################################################
#Import modules
import BaytesCore:
    BaytesCore,
    update!,
    infer,
    results,
    init,
    init!,
    propose,
    propose!,
    propagate,
    propagate!,
    result!,
    get_result,
    get_sym,
    generate_showvalues,
    generate,
    ResamplingMethod,
    shuffle!,
    resample!,
    weight!,
    jitter!

using BaytesCore:
    BaytesCore,
    AbstractAlgorithm,
    AbstractTune,
    AbstractConfiguration,
    AbstractDiagnostics,
    AbstractKernel,
    AbstractConstructor,
    Updater,
    Iterator,
    issmaller,
    computeESS,
    grab,
    to_NamedTuple,
    UpdateBool,
    UpdateTrue,
    UpdateFalse,
    update,
    split,
    ChainsTune,
    ParameterWeights,
    ResampleTune,
    ModelParameterBuffer,
    SampleDefault,
    ProposalTune,
    ValueHolder

using ModelWrappers:
    ModelWrappers,
    ModelWrapper,
    Tagged,
    Objective

using BaytesDiff:
    BaytesDiff,
    DiffObjective,
    ℓObjectiveResult,
    ℓDensityResult,
    ℓGradientResult

import ModelWrappers: predict, dynamics

import BaytesFilters: ParticleKernel
using BaytesMCMC, BaytesFilters, BaytesPMCMC

using DocStringExtensions:
    DocStringExtensions, TYPEDEF, TYPEDFIELDS, FIELDS, SIGNATURES, FUNCTIONNAME
using ArgCheck: ArgCheck, @argcheck
using UnPack: UnPack, @unpack, @pack!

using Random: Random, AbstractRNG, GLOBAL_RNG
using Statistics: Statistics, mean, std, sqrt, quantile, var, cor
#using Polyester: Polyester, @batch

############################################################################################
# Import sub-folder
include("Core/Core.jl")
include("Kernels/Kernels.jl")

############################################################################################
export
    # BaytesCore
    UpdateBool,
    UpdateTrue,
    UpdateFalse
    propose,
    propose!,
    propagate,
    propagate!,
    ResamplingMethod,
    infer,
    SampleDefault,
    generate,

    # SMC
    update!,
    init,
    init!,
    resample!,

    #ModelWrappers
    predict,
    dynamics

end
