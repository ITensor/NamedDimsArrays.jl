using NamedDimsArrays: NamedDimsArrays
using Test: @test, @testset
@testset "Test exports" begin
  exports = [:NamedDimsArrays, :NamedDimsArray, :aligndims, :named, :nameddimsarray]
  @test issetequal(names(NamedDimsArrays), exports)
end
