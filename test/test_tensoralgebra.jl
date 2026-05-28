using LinearAlgebra: LinearAlgebra, factorize, lq, norm, qr, svd
using NamedDimsArrays: NamedDimsArrays, dename, denamed, inds, namedoneto
using StableRNGs: StableRNG
using TensorAlgebra: TensorAlgebra, contract, gram_eigh_full, gram_eigh_full_with_pinv,
    left_null, left_orth, left_polar, matricize, orth, polar, right_null, right_orth,
    right_polar, unmatricize
using Test: @test, @test_broken, @testset

@testset "TensorAlgebra (eltype=$(elt))" for elt in
    (
        Float32,
        Float64,
        Complex{Float32},
        Complex{Float64},
    )
    @testset "contract" begin
        i = namedoneto(2, "i")
        j = namedoneto(2, "j")
        k = namedoneto(2, "k")
        na1 = randn(elt, i, j)
        na2 = randn(elt, j, k)
        na_dest = na1 * na2
        @test eltype(na_dest) ≡ elt
        @test dename(na_dest, (i, k)) ≈ denamed(na1) * denamed(na2)
    end
    @testset "matricize" begin
        i, j, k, l = namedoneto.((2, 3, 4, 5), ("i", "j", "k", "l"))
        na = randn(elt, i, j, k, l)
        na_fused = matricize(na, (k, i) => "a", (j, l) => "b")
        # Fuse all dimensions.
        @test dename(na_fused, ("a", "b")) ≈ reshape(
            dename(na, (k, i, j, l)),
            (
                denamed(length(k)) * denamed(length(i)),
                denamed(length(j)) * denamed(length(l)),
            )
        )
    end
    @testset "unmatricize" begin
        a, b = namedoneto.((6, 20), ("a", "b"))
        i, j, k, l = namedoneto.((2, 3, 4, 5), ("i", "j", "k", "l"))
        na = randn(elt, a, b)
        # Split all dimensions.
        na_split = unmatricize(na, "a" => (k, i), "b" => (j, l))
        @test dename(na_split, ("k", "i", "j", "l")) ≈
            reshape(
            dename(na, ("a", "b")),
            (denamed(k), denamed(i), denamed(j), denamed(l))
        )
    end
    @testset "Matrix functions" begin
        for f in NamedDimsArrays.MATRIX_FUNCTIONS
            f == :cbrt && elt <: Complex && continue
            f == :cbrt && VERSION < v"1.11-" && continue
            @eval begin
                i, j, k, l = namedoneto.((2, 2, 2, 2), ("i", "j", "k", "l"))
                rng = StableRNG(123)
                a = randn(rng, $elt, (i, j, k, l))
                fa = $f(a, (j, l), (k, i))
                m = dename(matricize(a, (j, l) => "a", (k, i) => "b"), ("a", "b"))
                fm = dename(matricize(fa, (j, l) => "a", (k, i) => "b"), ("a", "b"))
                @test fm ≈ $f(m)
            end
        end
    end
    @testset "qr/lq" begin
        dims = (2, 2, 2, 2)
        i, j, k, l = namedoneto.(dims, ("i", "j", "k", "l"))

        a = randn(elt, i, j)
        # TODO: Should this be allowed?
        # TODO: Add support for specifying new name.
        for f in
            (factorize, left_orth, left_polar, lq, orth, polar, qr, right_orth, right_polar)
            x, y = f(a, (i,))
            @test x * y ≈ a
        end

        a = randn(elt, i, j, k, l)
        # TODO: Add support for specifying new name.
        for f in
            (factorize, left_orth, left_polar, lq, orth, polar, qr, right_orth, right_polar)
            x, y = f(a, (i, k), (j, l))
            @test x * y ≈ a
        end
    end
    @testset "svd" begin
        dims = (2, 2, 2, 2)
        i, j, k, l = namedoneto.(dims, ("i", "j", "k", "l"))

        a = randn(elt, i, j)
        # TODO: Should this be allowed?
        # TODO: Add support for specifying new name.
        u, s, v = svd(a, (i,))
        @test u * s * v ≈ a

        a = randn(elt, i, j, k, l)
        # TODO: Add support for specifying new name.
        u, s, v = svd(a, (i, k), (j, l))
        @test u * s * v ≈ a

        # Test truncation.
        a = randn(elt, i, j, k, l)
        u, s, v = svd(a, (i, k), (j, l); trunc = (; maxrank = 2))
        @test u * s * v ≉ a
        @test Int.(Tuple(size(s))) == (2, 2)
    end
    @testset "left_null/right_null" begin
        dims = (2, 2, 2, 2)
        i, j, k, l = namedoneto.(dims, ("i", "j", "k", "l"))

        a = randn(elt, i, j, k, l)
        # TODO: Add support for specifying new name.
        for n in (left_null(a, (i, k), (j, l)), left_null(a, (i, k)))
            @test (i, k) ⊆ inds(n)
            @test norm(n * a) ≈ 0
        end
        for n in (right_null(a, (i, k), (j, l)), right_null(a, (i, k)))
            @test (j, l) ⊆ inds(n)
            @test norm(n * a) ≈ 0
        end
    end
    @testset "gram_eigh_full" begin
        # Build a Hermitian PSD a ≈ conj(b) * b over an aux dim, with codomain
        # (i, k) and domain (j, l) sharing the same axis lengths.
        i, j, k, l, aux = namedoneto.((2, 2, 2, 2, 5), ("i", "j", "k", "l", "aux"))
        b = randn(elt, aux, i, k)
        # conj(b) * b with the non-conjugated copy's (i, k) relabeled to
        # (j, l) to form the operator-shaped Hermitian a ≈ X' * X.
        b_dom = NamedDimsArrays.replacedimnames(b, "i" => "j", "k" => "l")
        a = conj(b) * b_dom

        for X in (gram_eigh_full(a, (i, k), (j, l)), gram_eigh_full(a, (i, k)))
            rank_name = only(setdiff(NamedDimsArrays.dimnames(X), ("i", "k")))
            X_dom = NamedDimsArrays.replacedimnames(X, "i" => "j", "k" => "l")
            @test (i, k) ⊆ inds(X)
            @test conj(X) * X_dom ≈ a
        end

        for (X, Y) in (
                gram_eigh_full_with_pinv(a, (i, k), (j, l)),
                gram_eigh_full_with_pinv(a, (i, k)),
            )
            rank_name = only(setdiff(NamedDimsArrays.dimnames(X), ("i", "k")))
            @test rank_name == only(setdiff(NamedDimsArrays.dimnames(Y), ("i", "k")))
            X_dom = NamedDimsArrays.replacedimnames(X, "i" => "j", "k" => "l")
            @test conj(X) * X_dom ≈ a
            # `X * Y` contracts the shared codomain names ((i, k)) and the
            # shared rank name, reducing to a scalar (the rank), so check the
            # matrix-level identity `X * Y ≈ I_rank` via parent storage.
            # `X` has axis order (rank, codomain...); `Y` has
            # (codomain..., rank).
            Xmat = reshape(dename(X, (rank_name, "i", "k")), :, 4)
            Ymat = reshape(dename(Y, ("i", "k", rank_name)), 4, :)
            @test Xmat * Ymat ≈ LinearAlgebra.I(size(Xmat, 1))
        end
    end
end
