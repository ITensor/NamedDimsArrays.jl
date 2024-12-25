using Test: @test, @test_throws, @testset
using NamedDimsArrays:
  NamedDimsArrays,
  AbstractNamedDimsArray,
  AbstractNamedDimsMatrix,
  Name,
  NameMismatch,
  NamedCartesianIndex,
  NamedCartesianIndices,
  NamedDimsArray,
  NamedDimsMatrix,
  aligndims,
  aligneddims,
  dename,
  denamed,
  dim,
  dimnames,
  dims,
  fusednames,
  isnamed,
  name,
  named,
  nameddims,
  replacedimnames,
  setdimnames,
  unname,
  unnamed

@testset "NamedDimsArrays.jl" begin
  @testset "Basic functionality" begin
    elt = Float64
    a = randn(elt, 3, 4)
    @test !isnamed(a)
    na = nameddims(a, ("i", "j"))
    @test na isa NamedDimsMatrix{elt,Matrix{elt}}
    @test na isa AbstractNamedDimsMatrix{elt}
    @test na isa NamedDimsArray{elt}
    @test na isa AbstractNamedDimsArray{elt}
    for na′ in (nameddims(na, ("j", "i")), NamedDimsArray(na, ("j", "i")))
      @test na′ isa NamedDimsMatrix{elt,<:PermutedDimsArray}
      @test dimnames(na′) == ("j", "i")
      @test na′ == na
    end
    @test_throws NameMismatch nameddims(na, ("j", "k"))
    @test_throws NameMismatch NamedDimsArray(na, ("j", "k"))
    @test_throws MethodError dename(a)
    @test_throws MethodError dename(a, ("i", "j"))
    @test_throws MethodError denamed(a, ("i", "j"))
    @test_throws MethodError unname(a, ("i", "j"))
    @test_throws MethodError unnamed(a, ("i", "j"))
    @test unname(a) == a
    @test dename(na) == a
    si, sj = size(na)
    ai, aj = axes(na)
    @test name(si) == "i"
    @test name(sj) == "j"
    @test name(ai) == "i"
    @test name(aj) == "j"
    @test isnamed(na)
    @test dimnames(na) == ("i", "j")
    @test dimnames(na, 1) == "i"
    @test dimnames(na, 2) == "j"
    @test dim(na, "i") == 1
    @test dim(na, "j") == 2
    @test dims(na, ("j", "i")) == (2, 1)
    @test na[1, 1] == a[1, 1]

    # getindex syntax
    i = Name("i")
    j = Name("j")
    @test a[i, j] == na
    @test @view(a[i, j]) == na
    @test na[j[1], i[2]] == a[2, 1]
    @test dimnames(na[j, i]) == ("j", "i")
    @test na[j, i] == na
    @test @view(na[j, i]) == na
    @test i[axes(a, 1)] == ai
    @test j[axes(a, 2)] == aj
    @test axes(na, i) == ai
    @test axes(na, j) == aj
    @test size(na, i) == si
    @test size(na, j) == sj

    # aliasing
    a′ = randn(2, 2)
    a′ij = @view a′[i, j]
    a′ij[i[1], j[2]] = 12
    @test a′ij[i[1], j[2]] == 12
    @test a′[1, 2] == 12
    a′ji = @view a′ij[j, i]
    a′ji[i[2], j[1]] = 21
    @test a′ji[i[2], j[1]] == 21
    @test a′ij[i[2], j[1]] == 21
    @test a′[2, 1] == 21

    a′ = randn(2, 2)
    a′ij = a′[i, j]
    a′ij[i[1], j[2]] = 12
    @test a′ij[i[1], j[2]] == 12
    @test a′[1, 2] ≠ 12
    a′ji = a′ij[j, i]
    a′ji[i[2], j[1]] = 21
    @test a′ji[i[2], j[1]] == 21
    @test a′ij[i[2], j[1]] ≠ 21
    @test a′[2, 1] ≠ 21

    a′ = dename(na)
    @test a′ isa Matrix{elt}
    @test a′ == a
    for a′ in (dename(na, ("j", "i")), unname(na, ("j", "i")))
      @test a′ isa Matrix{elt}
      @test a′ == a'
    end
    for a′ in (denamed(na, ("j", "i")), unnamed(na, ("j", "i")))
      @test a′ isa PermutedDimsArray{elt}
      @test a′ == a'
    end
    nb = setdimnames(na, ("k", "j"))
    @test dimnames(nb) == ("k", "j")
    @test dename(nb) == a
    nb = replacedimnames(na, "i" => "k")
    @test dimnames(nb) == ("k", "j")
    @test dename(nb) == a
    nb = replacedimnames(na, named(3, "i") => named(3, "k"))
    @test dimnames(nb) == ("k", "j")
    @test dename(nb) == a
    nb = replacedimnames(n -> n == "i" ? "k" : n, na)
    @test dimnames(nb) == ("k", "j")
    @test dename(nb) == a
    nb = setdimnames(na, named(3, "i") => named(3, "k"))
    na[1, 1] = 11
    @test na[1, 1] == 11
    @test size(na) == (named(3, "i"), named(4, "j"))
    @test length(na) == named(12, fusednames("i", "j"))
    @test axes(na) == (named(1:3, "i"), named(1:4, "j"))
    @test randn(named.((3, 4), ("i", "j"))) isa NamedDimsArray
    @test na["i" => 1, "j" => 2] == a[1, 2]
    @test na["j" => 2, "i" => 1] == a[1, 2]
    na["j" => 2, "i" => 1] = 12
    @test na[1, 2] == 12
    @test na[j => 1, i => 2] == a[2, 1]
    @test na[aj => 1, ai => 2] == a[2, 1]
    na[j => 1, i => 2] = 21
    @test na[2, 1] == 21
    na[aj => 1, ai => 2] = 2211
    @test na[2, 1] == 2211
    na′ = aligndims(na, ("j", "i"))
    @test unname(na′) isa Matrix{elt}
    @test a == permutedims(unname(na′), (2, 1))
    na′ = aligneddims(na, ("j", "i"))
    @test unname(na′) isa PermutedDimsArray{elt}
    @test a == permutedims(unname(na′), (2, 1))
    na′ = aligndims(na, (j, i))
    @test unname(na′) isa Matrix{elt}
    @test a == permutedims(unname(na′), (2, 1))
    na′ = aligneddims(na, (j, i))
    @test unname(na′) isa PermutedDimsArray{elt}
    @test a == permutedims(unname(na′), (2, 1))
    na′ = aligndims(na, (aj, ai))
    @test unname(na′) isa Matrix{elt}
    @test a == permutedims(unname(na′), (2, 1))
    na′ = aligneddims(na, (aj, ai))
    @test unname(na′) isa PermutedDimsArray{elt}
    @test a == permutedims(unname(na′), (2, 1))

    na = nameddims(randn(elt, 2, 3), (:i, :j))
    nb = nameddims(randn(elt, 3, 2), (:j, :i))
    nc = zeros(elt, named.((2, 3), (:i, :j)))
    Is = eachindex(na, nb)
    @test Is isa NamedCartesianIndices{2}
    @test issetequal(dimnames(Is), (:i, :j))
    for I in Is
      @test I isa NamedCartesianIndex{2}
      @test issetequal(name.(Tuple(I)), (:i, :j))
      nc[I] = na[I] + nb[I]
    end
    @test dename(nc, (:i, :j)) ≈ dename(na, (:i, :j)) + dename(nb, (:i, :j))

    a = nameddims(randn(elt, 2, 3), (:i, :j))
    b = nameddims(randn(elt, 3, 2), (:j, :i))
    c = a + b
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + dename(b, (:i, :j))
    c = a .+ b
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + dename(b, (:i, :j))
    c = map(+, a, b)
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + dename(b, (:i, :j))
    c = nameddims(Array{elt}(undef, 2, 3), (:i, :j))
    c = map!(+, c, a, b)
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + dename(b, (:i, :j))
    c = a .+ 2 .* b
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + 2 * dename(b, (:i, :j))
    c = nameddims(Array{elt}(undef, 2, 3), (:i, :j))
    c .= a .+ 2 .* b
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + 2 * dename(b, (:i, :j))
  end
  @testset "Shorthand constructors (eltype=$elt)" for elt in (
    Float32, ComplexF32, Float64, ComplexF64
  )
    i, j = named.((2, 2), ("i", "j"))
    value = rand(elt)
    for na in (zeros(elt, i, j), zeros(elt, (i, j)))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @test iszero(na)
    end
    for na in (fill(value, i, j), fill(value, (i, j)))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @test all(isequal(value), na)
    end
    for na in (rand(elt, i, j), rand(elt, (i, j)))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @test !iszero(na)
      @test all(x -> real(x) > 0, na)
    end
    for na in (randn(elt, i, j), randn(elt, (i, j)))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @test !iszero(na)
    end
  end
  @testset "Shorthand constructors (eltype=unspecified)" begin
    i, j = named.((2, 2), ("i", "j"))
    default_elt = Float64
    for na in (zeros(i, j), zeros((i, j)))
      @test eltype(na) === default_elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @test iszero(na)
    end
    for na in (rand(i, j), rand((i, j)))
      @test eltype(na) === default_elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @test !iszero(na)
      @test all(x -> real(x) > 0, na)
    end
    for na in (randn(i, j), randn((i, j)))
      @test eltype(na) === default_elt
      @test na isa NamedDimsArray
      @test dimnames(na) == ("i", "j")
      @test !iszero(na)
    end
  end
end
