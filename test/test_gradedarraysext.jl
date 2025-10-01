using GradedArrays: U1, dual, gradedrange, isdual
using NamedDimsArrays: dename, named
using Test: @test, @testset

@testset "NamedDimsArraysGradedArraysExt" begin
    r = gradedrange([U1(0) => 2, U1(1) => 2])
    nr = named(r, "i")
    nr_dual = dual(nr)
    @test isdual(nr_dual)
    @test isdual(dename(nr_dual))
end
