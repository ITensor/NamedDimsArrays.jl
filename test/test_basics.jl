using Combinatorics: Combinatorics
using NamedDimsArrays:
  NamedDimsArrays,
  AbstractNamedDimsArray,
  AbstractNamedDimsMatrix,
  NaiveOrderedSet,
  Name,
  NameMismatch,
  NamedDimsCartesianIndex,
  NamedDimsCartesianIndices,
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
  mapnameddimsindices,
  name,
  named,
  nameddimsarray,
  nameddimsindices,
  namedoneto,
  replacenameddimsindices,
  setnameddimsindices,
  unname,
  unnamed
using Test: @test, @test_throws, @testset

@testset "NamedDimsArrays.jl" begin
  @testset "Basic functionality" begin
    elt = Float64
    a = randn(elt, 3, 4)
    @test !isnamed(a)
    na = nameddimsarray(a, ("i", "j"))
    @test na isa NamedDimsMatrix{elt,Matrix{elt}}
    @test na isa AbstractNamedDimsMatrix{elt}
    @test na isa NamedDimsArray{elt}
    @test na isa AbstractNamedDimsArray{elt}
    @test_throws MethodError dename(a)
    @test_throws MethodError dename(a, ("i", "j"))
    @test_throws MethodError denamed(a, ("i", "j"))
    @test_throws MethodError unname(a, ("i", "j"))
    @test_throws MethodError unnamed(a, ("i", "j"))
    @test unname(a) == a
    @test dename(na) == a
    si, sj = size(na)
    ai, aj = axes(na)
    i = namedoneto(3, "i")
    j = namedoneto(4, "j")
    @test name(si) == i
    @test name(sj) == j
    @test name(ai) == i
    @test name(aj) == j
    @test isnamed(na)
    @test nameddimsindices(na) == (i, j)
    @test nameddimsindices(na, 1) == i
    @test nameddimsindices(na, 2) == j
    @test dimnames(na) == ("i", "j")
    @test dimnames(na, 1) == "i"
    @test dimnames(na, 2) == "j"
    @test dim(na, "i") == 1
    @test dim(na, "j") == 2
    @test dims(na, ("j", "i")) == (2, 1)
    @test na[1, 1] == a[1, 1]

    @test_throws ErrorException NamedDimsArray(randn(4), namedoneto.((2, 2), ("i", "j")))
    @test_throws ErrorException NamedDimsArray(randn(2, 2), namedoneto.((2, 3), ("i", "j")))

    a = randn(elt, 3, 4)
    na = nameddimsarray(a, ("i", "j"))
    a′ = Array(na)
    @test eltype(a′) === elt
    @test a′ isa Matrix{elt}
    @test a′ == a

    a = randn(elt, 3, 4)
    na = nameddimsarray(a, ("i", "j"))
    for a′ in (Array{Float32}(na), Matrix{Float32}(na))
      @test eltype(a′) === Float32
      @test a′ isa Matrix{Float32}
      @test a′ == Float32.(a)
    end

    a = randn(elt, 2, 2, 2)
    na = nameddimsarray(a, ("i", "j", "k"))
    b = randn(elt, 2, 2, 2)
    nb = nameddimsarray(b, ("k", "i", "j"))
    copyto!(na, nb)
    @test na == nb
    @test dename(na) == dename(nb, ("i", "j", "k"))
    @test dename(na) == permutedims(dename(nb), (2, 3, 1))

    a = randn(elt, 3, 4)
    na = nameddimsarray(a, ("i", "j"))
    i = namedoneto(3, "i")
    j = namedoneto(4, "j")
    ai, aj = axes(na)
    for na′ in (
      similar(na, Float32, (j, i)),
      similar(na, Float32, NaiveOrderedSet((j, i))),
      similar(na, Float32, (aj, ai)),
      similar(na, Float32, NaiveOrderedSet((aj, ai))),
      similar(a, Float32, (j, i)),
      similar(a, Float32, NaiveOrderedSet((j, i))),
      similar(a, Float32, (aj, ai)),
      similar(a, Float32, NaiveOrderedSet((aj, ai))),
    )
      @test eltype(na′) === Float32
      @test all(nameddimsindices(na′) .== (j, i))
      @test na′ ≠ na
    end

    a = randn(elt, 3, 4)
    na = nameddimsarray(a, ("i", "j"))
    i = namedoneto(3, "i")
    j = namedoneto(4, "j")
    ai, aj = axes(na)
    for na′ in (
      similar(na, (j, i)),
      similar(na, NaiveOrderedSet((j, i))),
      similar(na, (aj, ai)),
      similar(na, NaiveOrderedSet((aj, ai))),
      similar(a, (j, i)),
      similar(a, NaiveOrderedSet((j, i))),
      similar(a, (aj, ai)),
      similar(a, NaiveOrderedSet((aj, ai))),
    )
      @test eltype(na′) === eltype(na)
      @test all(nameddimsindices(na′) .== (j, i))
      @test na′ ≠ na
    end

    # getindex syntax
    i = Name("i")
    j = Name("j")
    @test a[i, j] == na
    @test @view(a[i, j]) == na
    @test na[j[1], i[2]] == a[2, 1]
    @test nameddimsindices(na[j, i]) == (named(1:3, "i"), named(1:4, "j"))
    @test na[j, i] == na
    @test @view(na[j, i]) == na
    @test i[axes(a, 1)] == named(1:3, "i")
    @test j[axes(a, 2)] == named(1:4, "j")
    @test axes(na, i) == ai
    @test axes(na, j) == aj
    @test size(na, i) == si
    @test size(na, j) == sj

    # Regression test for ambiguity error with
    # `Base.getindex(A::Array, I::AbstractUnitRange{<:Integer})`.
    i = namedoneto(2, "i")
    a = randn(elt, 2)
    na = a[i]
    @test na isa NamedDimsArray{elt}
    @test dimnames(na) == ("i",)
    @test dename(na) == a

    # aliasing
    a′ = randn(elt, 2, 2)
    i = Name("i")
    j = Name("j")
    a′ij = @view a′[i, j]
    a′ij[i[1], j[2]] = 12
    @test a′ij[i[1], j[2]] == 12
    @test a′[1, 2] == 12
    a′ji = @view a′ij[j, i]
    a′ji[i[2], j[1]] = 21
    @test a′ji[i[2], j[1]] == 21
    @test a′ij[i[2], j[1]] == 21
    @test a′[2, 1] == 21

    a′ = randn(elt, 2, 2)
    i = Name("i")
    j = Name("j")
    a′ij = a′[i, j]
    a′ij[i[1], j[2]] = 12
    @test a′ij[i[1], j[2]] == 12
    @test a′[1, 2] ≠ 12
    a′ji = a′ij[j, i]
    a′ji[i[2], j[1]] = 21
    @test a′ji[i[2], j[1]] == 21
    @test a′ij[i[2], j[1]] ≠ 21
    @test a′[2, 1] ≠ 21

    a = randn(elt, 3, 4)
    na = nameddimsarray(a, ("i", "j"))
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
    nb = setnameddimsindices(na, ("k", "j"))
    @test nameddimsindices(nb) == (named(1:3, "k"), named(1:4, "j"))
    @test dename(nb) == a
    nb = replacenameddimsindices(na, "i" => "k")
    @test nameddimsindices(nb) == (named(1:3, "k"), named(1:4, "j"))
    @test dename(nb) == a
    nb = replacenameddimsindices(na, named(1:3, "i") => named(1:3, "k"))
    @test nameddimsindices(nb) == (named(1:3, "k"), named(1:4, "j"))
    @test dename(nb) == a
    nb = replacenameddimsindices(n -> n == named(1:3, "i") ? named(1:3, "k") : n, na)
    @test nameddimsindices(nb) == (named(1:3, "k"), named(1:4, "j"))
    @test dename(nb) == a
    nb = mapnameddimsindices(n -> n == named(1:3, "i") ? named(1:3, "k") : n, na)
    @test nameddimsindices(nb) == (named(1:3, "k"), named(1:4, "j"))
    @test dename(nb) == a
    nb = setnameddimsindices(na, named(3, "i") => named(3, "k"))
    na[1, 1] = 11
    @test na[1, 1] == 11
    @test Tuple(size(na)) == (named(3, named(1:3, "i")), named(4, named(1:4, "j")))
    @test length(na) == named(12, fusednames(named(1:3, "i"), named(1:4, "j")))
    @test Tuple(axes(na)) == (named(1:3, named(1:3, "i")), named(1:4, named(1:4, "j")))
    @test randn(named.((3, 4), ("i", "j"))) isa NamedDimsArray
    @test na["i" => 1, "j" => 2] == a[1, 2]
    @test na["j" => 2, "i" => 1] == a[1, 2]
    na["j" => 2, "i" => 1] = 12
    @test na[1, 2] == 12
    @test na[j => 1, i => 2] == a[2, 1]
    na[j => 1, i => 2] = 21
    @test na[2, 1] == 21
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

    na = nameddimsarray(randn(elt, 2, 3), (:i, :j))
    nb = nameddimsarray(randn(elt, 3, 2), (:j, :i))
    nc = zeros(elt, named.((2, 3), (:i, :j)))
    Is = eachindex(na, nb)
    @test Is isa NamedDimsCartesianIndices{2}
    @test issetequal(nameddimsindices(Is), (named(1:2, :i), named(1:3, :j)))
    for I in Is
      @test I isa NamedDimsCartesianIndex{2}
      @test issetequal(name.(Tuple(I)), (:i, :j))
      nc[I] = na[I] + nb[I]
    end
    @test dename(nc, (:i, :j)) ≈ dename(na, (:i, :j)) + dename(nb, (:i, :j))

    a = nameddimsarray(randn(elt, 2, 3), (:i, :j))
    b = nameddimsarray(randn(elt, 3, 2), (:j, :i))
    c = a + b
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + dename(b, (:i, :j))
    c = a .+ b
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + dename(b, (:i, :j))
    c = map(+, a, b)
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + dename(b, (:i, :j))
    c = nameddimsarray(Array{elt}(undef, 2, 3), (:i, :j))
    c = map!(+, c, a, b)
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + dename(b, (:i, :j))
    c = a .+ 2 .* b
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + 2 * dename(b, (:i, :j))
    c = nameddimsarray(Array{elt}(undef, 2, 3), (:i, :j))
    c .= a .+ 2 .* b
    @test dename(c, (:i, :j)) ≈ dename(a, (:i, :j)) + 2 * dename(b, (:i, :j))

    # Regression test for proper permutations.
    a = nameddimsarray(randn(elt, 2, 3, 4), (:i, :j, :k))
    I = (:i => 2, :j => 3, :k => 4)
    for I′ in Combinatorics.permutations(I)
      @test a[I′...] == a[2, 3, 4]
      a′ = copy(a)
      a′[I′...] = zero(Bool)
      @test iszero(a′[2, 3, 4])
    end
    I = (:i => 2, :j => 2:3, :k => 4)
    for I′ in Combinatorics.permutations(I)
      @test a[I′...] == a[2, 2:3, 4]
      ## TODO: This is broken, investigate.
      ## a′[I′...] = zeros(Bool, 2)
      ## @test iszero(a′[2, 2:3, 4])
    end
  end
  @testset "Shorthand constructors (eltype=$elt)" for elt in (
    Float32, ComplexF32, Float64, ComplexF64
  )
    i, j = named.((2, 2), ("i", "j"))
    value = rand(elt)
    for na in (zeros(elt, i, j), zeros(elt, (i, j)))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test nameddimsindices(na) == Base.oneto.((i, j))
      @test iszero(na)
    end
    for na in (fill(value, i, j), fill(value, (i, j)))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test nameddimsindices(na) == Base.oneto.((i, j))
      @test all(isequal(value), na)
    end
    for na in (rand(elt, i, j), rand(elt, (i, j)))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test nameddimsindices(na) == Base.oneto.((i, j))
      @test !iszero(na)
      @test all(x -> real(x) > 0, na)
    end
    for na in (randn(elt, i, j), randn(elt, (i, j)))
      @test eltype(na) === elt
      @test na isa NamedDimsArray
      @test nameddimsindices(na) == Base.oneto.((i, j))
      @test !iszero(na)
    end
  end
  @testset "Shorthand constructors (eltype=unspecified)" begin
    i, j = named.((2, 2), ("i", "j"))
    default_elt = Float64
    for na in (zeros(i, j), zeros((i, j)))
      @test eltype(na) === default_elt
      @test na isa NamedDimsArray
      @test nameddimsindices(na) == Base.oneto.((i, j))
      @test iszero(na)
    end
    for na in (rand(i, j), rand((i, j)))
      @test eltype(na) === default_elt
      @test na isa NamedDimsArray
      @test nameddimsindices(na) == Base.oneto.((i, j))
      @test !iszero(na)
      @test all(x -> real(x) > 0, na)
    end
    for na in (randn(i, j), randn((i, j)))
      @test eltype(na) === default_elt
      @test na isa NamedDimsArray
      @test nameddimsindices(na) == Base.oneto.((i, j))
      @test !iszero(na)
    end
  end
  @testset "NaiveOrderedSet" begin
    # Broadcasting
    s = NaiveOrderedSet((1, 2))
    @test s .+ [3, 4] == [4, 6]
    @test s .+ (3, 4) === (4, 6)

    s = NaiveOrderedSet(("a", "b", "c"))
    @test all(s .== ("a", "b", "c"))
    @test values(s) == ("a", "b", "c")
    @test Tuple(s) == ("a", "b", "c")
    @test s[1] == "a"
    @test s[2] == "b"
    @test s[3] == "c"
    for s′ in (
      replace(x -> x == "b" ? "x" : x, s),
      replace(s, "b" => "x"),
      map(x -> x == "b" ? "x" : x, s),
    )
      @test s′ isa NaiveOrderedSet
      @test Tuple(s′) == ("a", "x", "c")
      @test s′[1] == "a"
      @test s′[2] == "x"
      @test s′[3] == "c"
    end
  end
end
