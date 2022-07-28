############################################################################################
# Import External Packages
using Test
using Random: Random, AbstractRNG, seed!
using UnPack: UnPack, @unpack, @pack!
using Distributions

############################################################################################
# Import Baytes Packages
using
    BaytesCore,
    ModelWrappers,
    BaytesMCMC,
    BaytesFilters,
    BaytesPMCMC,
    BaytesSMC
#using .BaytesSMC
############################################################################################
# Include Files
include("testhelper/TestHelper.jl")

############################################################################################
# Run Tests
@testset "All tests" begin
    include("test-utility.jl")
    include("test-construction.jl")
end
