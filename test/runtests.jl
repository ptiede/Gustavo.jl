using Gustavo
using Test

@testset "Gustavo.jl" begin
    @test isdefined(Gustavo, :Bandpass)
    @test Gustavo.Bandpass isa Module
    @test isdefined(Gustavo.Bandpass, :solve_bandpass)
end
