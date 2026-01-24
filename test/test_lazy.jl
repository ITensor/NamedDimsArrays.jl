using NamedDimsArrays: NamedDimsArrays as NDA, Name, nameddims
using TensorAlgebra: +ₗ, *ₗ, conjed
using Test: @test, @testset

@testset "Lazy named dims arrays" begin
    i, j, k = Name.((:i, :j, :k))
    a = randn(ComplexF64, 2, 2)[i, j]
    b = randn(ComplexF64, 2, 2)[j, i]
    c = randn(ComplexF64, 2, 2)[j, k]
    d = randn(ComplexF64, 2, 2)[i, k]

    x = 2 *ₗ a
    @test x ≡ NDA.ScaledNamedDimsArray(2, a)
    @test copy(x) ≈ 2a

    x = conjed(a)
    @test x ≡ NDA.ConjNamedDimsArray(a)
    @test copy(x) ≈ conj(a)

    x = a +ₗ b
    @test x ≡ NDA.AddNamedDimsArray(a, b)
    ## TODO: FIXME ## @test copy(x) ≈ a + b

    x = a *ₗ c
    @test x ≡ NDA.MulNamedDimsArray(a, c)
    @test copy(x) ≈ a * c
end
