using NamedDimsArrays: AbstractNamedInteger, NamedInteger, dename, named, name
using Test: @test, @testset

@testset "Named integer" begin
    i = named(3, :i)
    @test i isa NamedInteger
    @test i isa AbstractNamedInteger
    @test dename(i) ≡ 3
    @test name(i) ≡ :i
    for T in (Int32, Int64, Float32, Float64)
        @test T(i) ≡ T(3)
        @test convert(T, i) ≡ T(3)
    end
end
