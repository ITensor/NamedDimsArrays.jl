using LinearAlgebra: qr, svd
using NamedDimsArrays: namedoneto, dename
using TensorAlgebra: TensorAlgebra, contract, fusedims, splitdims
using Test: @test, @testset, @test_broken
elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "TensorAlgebra (eltype=$(elt))" for elt in elts
  @testset "contract" begin
    i = namedoneto(2, "i")
    j = namedoneto(2, "j")
    k = namedoneto(2, "k")
    na1 = randn(elt, i, j)
    na2 = randn(elt, j, k)
    na_dest = contract(na1, na2)
    @test eltype(na_dest) === elt
    @test dename(na_dest, (i, k)) ≈ dename(na1) * dename(na2)
  end
  @testset "fusedims" begin
    i, j, k, l = namedoneto.((2, 3, 4, 5), ("i", "j", "k", "l"))
    na = randn(elt, i, j, k, l)
    na_fused = fusedims(na, (k, i) => "a", (j, l) => "b")
    # Fuse all dimensions.
    @test dename(na_fused, ("a", "b")) ≈ reshape(
      dename(na, (k, i, j, l)),
      (dename(length(k)) * dename(length(i)), dename(length(j)) * dename(length(l))),
    )
    na_fused = fusedims(na, (k, i) => "a")
    # Fuse a subset of dimensions.
    @test dename(na_fused, ("a", "j", "l")) ≈ reshape(
      dename(na, (k, i, j, l)),
      (dename(length(k)) * dename(length(i)), dename(length(j)), dename(length(l))),
    )
  end
  @testset "splitdims" begin
    a, b = namedoneto.((6, 20), ("a", "b"))
    i, j, k, l = namedoneto.((2, 3, 4, 5), ("i", "j", "k", "l"))
    na = randn(elt, a, b)
    # Split all dimensions.
    na_split = splitdims(na, "a" => (k, i), "b" => (j, l))
    @test dename(na_split, ("k", "i", "j", "l")) ≈
      reshape(dename(na, ("a", "b")), (dename(k), dename(i), dename(j), dename(l)))
    # Split a subset of dimensions.
    na_split = splitdims(na, "a" => (j, i))
    @test dename(na_split, ("j", "i", "b")) ≈
      reshape(dename(na, ("a", "b")), (dename(j), dename(i), dename(b)))
  end
  @testset "qr" begin
    dims = (2, 2, 2, 2)
    i, j, k, l = namedoneto.(dims, ("i", "j", "k", "l"))

    a = randn(elt, i, j)
    # TODO: Should this be allowed?
    # TODO: Add support for specifying new name.
    q, r = qr(a, (i,))
    @test q * r ≈ a

    a = randn(elt, i, j, k, l)
    # TODO: Add support for specifying new name.
    q, r = qr(a, (i, k), (j, l))
    @test q * r ≈ a
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
  end
end