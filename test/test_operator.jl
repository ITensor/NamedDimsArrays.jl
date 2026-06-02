using LinearAlgebra: I, norm
using NamedDimsArrays: NamedDimsArrays as NDA, NamedDimsArray, NamedDimsOperator, apply,
    denamed, dimnames, nameddims, namedoneto, operator, product, replacedimnames, state
using TensorAlgebra: gram_eigh_full, gram_eigh_full_with_pinv
using Test: @test, @testset

@testset "operator" begin
    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    @test o isa NamedDimsOperator{Float64}
    @test eltype(o) ≡ Float64
    @test issetequal(NDA.codomainnames(o), ("i'", "j'"))
    @test issetequal(NDA.domainnames(o), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    õ = similar(o)
    @test õ isa NamedDimsOperator{Float64}
    @test eltype(õ) ≡ Float64
    @test issetequal(NDA.codomainnames(õ), ("i'", "j'"))
    @test issetequal(NDA.domainnames(õ), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    õ = similar(o, Float32)
    @test õ isa NamedDimsOperator{Float32}
    @test eltype(õ) ≡ Float32
    @test issetequal(NDA.codomainnames(õ), ("i'", "j'"))
    @test issetequal(NDA.domainnames(õ), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    @test o isa NamedDimsOperator
    o² = product(o, o)
    @test issetequal(dimnames(o²), ("i'", "j'", "i", "j"))
    õ = replacedimnames(
        state(o), "i" => "i'", "j" => "j'", "i'" => "x", "j'" => "y"
    )
    o²′ = replacedimnames(õ * o, "x" => "i'", "y" => "j'")
    @test state(o²) ≈ o²′

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    v = NamedDimsArray(randn(2, 2), ("i", "j"))
    ov = apply(o, v)
    @test issetequal(dimnames(ov), ("i", "j"))
    @test ov ≈ replacedimnames(o * v, "i'" => "i", "j'" => "j")
end

@testset "gram_eigh_full on AbstractNamedDimsOperator" begin
    n = 5
    B = randn(n, n)
    A = B * B'  # Hermitian PSD
    M_nda = nameddims(A, ("ket", "bra"))
    M_op = operator(M_nda, ["ket"], ["bra"])

    X_op = gram_eigh_full(M_op)
    X_arr = gram_eigh_full(M_nda, ("ket",), ("bra",))
    # Operator entry forwards to the named-array entry: same data, same shape.
    @test size(parent(X_op)) == size(parent(X_arr))

    Xp = parent(X_op)
    @test Xp * Xp' ≈ A

    X2, Y2 = gram_eigh_full_with_pinv(M_op)
    Xp2 = parent(X2)
    Yp2 = parent(Y2)
    @test Xp2 * Xp2' ≈ A
    @test Yp2 * Xp2 ≈ I(n)
end
