using LinearAlgebra: qr
using NamedDimsArrays: named, dename
using TensorAlgebra: TensorAlgebra, contract, fusedims, splitdims
using Test: @test, @testset, @test_broken
elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "TensorAlgebra (eltype=$(elt))" for elt in elts
  @testset "contract" begin
    i = named(2, "i")
    j = named(2, "j")
    k = named(2, "k")
    na1 = randn(elt, i, j)
    na2 = randn(elt, j, k)
    na_dest = contract(na1, na2)
    @test eltype(na_dest) === elt
    @test dename(na_dest, (i, k)) ≈ dename(na1) * dename(na2)
  end
  @testset "fusedims" begin
    i, j, k, l = named.((2, 3, 4, 5), ("i", "j", "k", "l"))
    na = randn(elt, i, j, k, l)
    na_fused = fusedims(na, (k, i) => "a", (j, l) => "b")
    # Fuse all dimensions.
    @test dename(na_fused, ("a", "b")) ≈
      reshape(dename(na, (k, i, j, l)), (dename(k) * dename(i), dename(j) * dename(l)))
    na_fused = fusedims(na, (k, i) => "a")
    # Fuse a subset of dimensions.
    @test dename(na_fused, ("a", "j", "l")) ≈
      reshape(dename(na, (k, i, j, l)), (dename(k) * dename(i), dename(j), dename(l)))
  end
  @testset "splitdims" begin
    a, b = named.((6, 20), ("a", "b"))
    i, j, k, l = named.((2, 3, 4, 5), ("i", "j", "k", "l"))
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
    i, j, k, l = named.(dims, ("i", "j", "k", "l"))

    na = randn(elt, i, j)
    # TODO: Should this be allowed?
    # TODO: Add support for specifying new name.
    q, r = qr(na, (i,))
    @test q * r ≈ na

    na = randn(elt, i, j, k, l)
    # TODO: Add support for specifying new name.
    q, r = qr(na, (i, k), (j, l))
    @test contract(q, r) ≈ na
  end
end
