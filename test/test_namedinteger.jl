using NamedDimsArrays: AbstractNamedInteger, NamedInteger, dename, named, name
using Test: @test, @testset

@testset "Named integer" begin
    i = named(3, :i)
    @test i isa NamedInteger
    @test i isa AbstractNamedInteger
    @test dename(i) ≡ 3
    @test name(i) ≡ :i
    for type in (Int32, Int64, Float32, Float64)
        @test type(i) ≡ type(3)
        @test convert(type, i) ≡ type(3)
    end
end
