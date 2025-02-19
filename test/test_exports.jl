using NamedDimsArrays: NamedDimsArrays
using Test: @test, @testset
@testset "Test exports" begin
  exports = [
    :NamedDimsArrays,
    :NamedDimsArray,
    :aligndims,
    :named,
    :nameddimsarray,
  ]
  public = [:to_nameddimsindices]
  if VERSION ≥ v"1.11-"
    exports = [exports; public]
  end
  @test issetequal(names(NamedDimsArrays), exports)
end
