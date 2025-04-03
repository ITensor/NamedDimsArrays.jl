using Adapt: adapt
using NamedDimsArrays: nameddimsarray
using Test: @test, @testset

@testset "Adapt (eltype=$elt)" for elt in
                                   (Float32, Float64, Complex{Float32}, Complex{Float64})
  na = nameddimsarray(randn(2, 2), ("i", "j"))
  na_complex = adapt(Array{complex(elt)}, na)
  @test na ≈ na_complex
  @test eltype(na_complex) === complex(elt)
end
