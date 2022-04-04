############################################################################################
@testset "Batchsize computation: " begin
    Nchains = 10
    Nthreads = 11
    batchsize = BaytesSMC.compute_batchsize(Nchains, Nthreads)
    @test batchsize == 1

    Nchains = 11
    Nthreads = 10
    batchsize = BaytesSMC.compute_batchsize(Nchains, Nthreads)
    @test batchsize == 2

    Nchains = 10
    Nthreads = 10
    batchsize = BaytesSMC.compute_batchsize(Nchains, Nthreads)
    @test batchsize == 1
end
