using LinearAlgebra: I, norm
using NamedDimsArrays: NamedDimsArrays as NDA, NamedDimsArray, NamedDimsOperator, apply,
    codomainnames, dename, denamed, dimnames, domainnames, nameddims, namedoneto, operator,
    product, replacedimnames, similar_operator, state
using Random: Random
using StableRNGs: StableRNG
using TensorAlgebra: gram_eigh_full, gram_eigh_full_with_pinv, matricize
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

@testset "one(::AbstractNamedDimsOperator)" begin
    # Identity-operator construction: matricized form is the identity matrix.
    i, j, k, l = namedoneto.((2, 3, 2, 3), ("i", "j", "k", "l"))
    op = operator(randn(i, j, k, l), ("i", "j"), ("k", "l"))
    Id = one(op)
    @test Id isa NamedDimsOperator{Float64}
    @test codomainnames(Id) == codomainnames(op)
    @test domainnames(Id) == domainnames(op)
    Id_mat = matricize(state(Id), (i, j) => "row", (k, l) => "col")
    @test dename(Id_mat, ("row", "col")) ≈ I(6)
end

@testset "one(::AbstractNamedDimsArray, codomain, domain)" begin
    # Trivial codomain/domain layout.
    i, j, k, l = namedoneto.((2, 3, 2, 3), ("i", "j", "k", "l"))
    a = randn(i, j, k, l)
    Id = one(a, (i, j), (k, l))
    Id_mat = matricize(Id, (i, j) => "row", (k, l) => "col")
    @test dename(Id_mat, ("row", "col")) ≈ I(6)

    # Non-trivial axis ordering: codomain/domain are interleaved in `a`.
    p, q, r, s = namedoneto.((2, 4, 2, 4), ("p", "q", "r", "s"))
    a = randn(p, r, q, s)  # storage order interleaves codomain (p, q) and domain (r, s)
    Id = one(a, (p, q), (r, s))
    @test issetequal(dimnames(Id), ("p", "r", "q", "s"))
    Id_mat = matricize(Id, (p, q) => "row", (r, s) => "col")
    @test dename(Id_mat, ("row", "col")) ≈ I(8)
end

@testset "similar_operator" begin
    # Five-arg canonical: explicit element type, axes, codomain, domain names.
    op = similar_operator(randn(3, 3), Float32, (Base.OneTo(3),), ("i'",), ("i",))
    @test op isa NamedDimsOperator{Float32}
    @test issetequal(codomainnames(op), ("i'",))
    @test issetequal(domainnames(op), ("i",))

    # Codomain names default to fresh `randname` outputs.
    op = similar_operator(randn(3, 3), Float64, (Base.OneTo(3),), ("i",))
    @test op isa NamedDimsOperator{Float64}
    @test issetequal(domainnames(op), ("i",))
    @test only(codomainnames(op)) != "i"

    # Named-axes form reuses each axis's name as the domain.
    i = namedoneto(3, "i")
    op = similar_operator(randn(3, 3), Float64, (i,))
    @test issetequal(domainnames(op), ("i",))
    @test only(codomainnames(op)) != "i"

    # Element type defaults to `eltype(prototype)`.
    op = similar_operator(randn(ComplexF32, 3, 3), (Base.OneTo(3),), ("i'",), ("i",))
    @test eltype(op) === ComplexF32
end

@testset "randn!(::AbstractNamedDimsOperator) / rand!" begin
    op = operator(zeros(3, 3), ("i'",), ("i",))
    rng = StableRNG(123)
    Random.randn!(rng, op)
    @test !all(iszero, denamed(state(op)))

    Random.rand!(rng, op)
    @test !all(iszero, denamed(state(op)))
    @test all(0 .≤ denamed(state(op)) .≤ 1)
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
