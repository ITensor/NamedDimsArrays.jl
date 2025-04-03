using NamedDimsArrays: NamedDimsArrays
using Test: @test, @testset
@testset "Test exports" begin
  exports = [:NamedDimsArrays, :NamedDimsArray, :aligndims, :named, :nameddimsarray]
  publics = [:to_nameddimsindices]
  if VERSION ≥ v"1.11-"
    exports = [exports; publics]
  end
  @test issetequal(names(NamedDimsArrays), exports)
end
