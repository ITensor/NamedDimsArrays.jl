import LinearAlgebra as LA
using NamedDimsArrays: denamed, dimnames, named
using Test: @test, @testset

@testset "LinearAlgebra (eltype=$(elt))" for elt in
    (Float32, Float64, Complex{Float32})
    i, j = named.(2, (:i, :j))
    a = randn(elt, i, j)
    b = randn(elt, j, i)
    @test LA.norm(a) ≈ LA.norm(denamed(a))
    @test denamed(LA.normalize(a)) ≈ LA.normalize(denamed(a))
    @test denamed(LA.normalize!(copy(a))) ≈ LA.normalize(denamed(a))
    @test denamed(LA.rmul!(copy(a), 2)) ≈ 2 * denamed(a)
    @test denamed(LA.lmul!(2, copy(a))) ≈ 2 * denamed(a)
    @test denamed(LA.rdiv!(copy(a), 2)) ≈ denamed(a) / 2
    @test denamed(LA.ldiv!(2, copy(a))) ≈ 2 \ denamed(a)
    @test LA.dot(a, b) ≈ LA.dot(denamed(a), denamed(b, dimnames(a)))
end
