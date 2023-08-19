############################################################################################
# Import External Packages
using Test
using Random: Random, AbstractRNG, seed!
using SimpleUnPack: SimpleUnPack, @unpack, @pack!
using Distributions

############################################################################################
# Import Baytes Packages
using
    BaytesCore,
    ModelWrappers,
    BaytesMCMC,
    BaytesFilters,
    BaytesPMCMC,
    BaytesOptim,
    BaytesSMC

############################################################################################
# Include Files
include("testhelper/TestHelper.jl")

############################################################################################
# Run Tests
@testset "All tests" begin
    include("test-utility.jl")
    include("test-construction.jl")
end
