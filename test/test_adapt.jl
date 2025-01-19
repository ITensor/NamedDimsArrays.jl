using Adapt: adapt
using NamedDimsArrays: nameddims
using Test: @test, @testset

using NamedDimsArrays: nameddimsindices

@testset "Adapt (eltype=$elt)" for elt in
                                   (Float32, Float64, Complex{Float32}, Complex{Float64})
  na = nameddims(randn(2, 2), ("i", "j"))
  na_complex = adapt(Array{complex(elt)}, na)
  @test na ≈ na_complex
  @test eltype(na_complex) === complex(elt)
end
