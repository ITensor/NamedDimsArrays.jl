using Combinatorics: Combinatorics
import NamedDimsArrays as NDA
using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedDimsMatrix, LittleSet,
    Name, NameMismatch, NamedDimsCartesianIndex, NamedDimsCartesianIndices, NamedDimsArray,
    NamedDimsMatrix, NamedDimsOperator
using NamedDimsArrays: aligndims, aligneddims, apply, dename, denamed, dim, dimnames, dims,
    fusednames, isnamed, mapinds, name, named, nameddims, inds, namedoneto, operator,
    product, replaceinds, setinds, state, @names
using Test: @test, @test_throws, @testset
using VectorInterface: scalartype

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "NamedDimsArrays.jl" begin
    @testset "Basic functionality (eltype=$elt)" for elt in elts
        a = randn(elt, 3, 4)
        @test !isnamed(a)
        na = nameddims(a, ("i", "j"))
        @test na isa NamedDimsMatrix{elt, Matrix{elt}}
        @test na isa AbstractNamedDimsMatrix{elt}
        @test na isa NamedDimsArray{elt}
        @test na isa AbstractNamedDimsArray{elt}
        @test_throws MethodError denamed(a)
        @test_throws MethodError dename(a, ("i", "j"))
        @test_throws MethodError denamed(a, ("i", "j"))
        @test denamed(na) == a
        si, sj = size(na)
        ai, aj = axes(na)
        i = namedoneto(3, "i")
        j = namedoneto(4, "j")
        @test name(si) == "i"
        @test name(sj) == "j"
        @test name(ai) == "i"
        @test name(aj) == "j"
        @test isnamed(na)
        @test inds(na) == (i, j)
        @test inds(na, 1) == i
        @test inds(na, 2) == j
        @test dimnames(na) == ("i", "j")
        @test dimnames(na, 1) == "i"
        @test dimnames(na, 2) == "j"
        @test dim(na, "i") == 1
        @test dim(na, "j") == 2
        @test dims(na, ("j", "i")) == (2, 1)
        @test na[1, 1] == a[1, 1]

        # equals (==)/isequal
        a = randn(elt, 3, 4)
        na = nameddims(a, ("i", "j"))
        @test na == na
        @test na == aligndims(na, ("j", "i"))
        @test isequal(na, na)
        @test isequal(na, aligndims(na, ("j", "i")))
        @test hash(na) == hash(aligndims(na, ("j", "i")))
        # Regression test that NamedDimsArrays
        # with different names are not equal (as opposed to
        # erroring).
        @test na ≠ nameddims(a, ("j", "k"))
        @test !isequal(na, nameddims(a, ("j", "k")))
        @test hash(na) ≠ hash(nameddims(a, ("j", "k")))

        a = randn(elt, 2, 2)
        na = nameddims(a, ("i", "j"))
        @test CartesianIndices(na) == CartesianIndices(a)
        @test collect(pairs(na)) == (CartesianIndices(a) .=> a)

        @test_throws ArgumentError NamedDimsArray(randn(4), namedoneto.((2, 2), ("i", "j")))
        @test_throws ErrorException NamedDimsArray(randn(2, 2), namedoneto.((2, 3), ("i", "j")))

        a = randn(elt, 3, 4)
        na = nameddims(a, ("i", "j"))
        @test eltype(na) ≡ elt
        @test scalartype(na) ≡ elt
        a′ = Array(na)
        @test eltype(a′) ≡ elt
        @test a′ isa Matrix{elt}
        @test a′ == a

        if elt <: Real
            a = randn(elt, 3, 4)
            na = nameddims(a, ("i", "j"))
            for a′ in (Array{Float32}(na), Matrix{Float32}(na))
                @test eltype(a′) ≡ Float32
                @test a′ isa Matrix{Float32}
                @test a′ == Float32.(a)
            end
        end

        a = randn(elt, 2, 2, 2)
        na = nameddims(a, ("i", "j", "k"))
        b = randn(elt, 2, 2, 2)
        nb = nameddims(b, ("k", "i", "j"))
        copyto!(na, nb)
        @test na == nb
        @test denamed(na) == dename(nb, ("i", "j", "k"))
        @test denamed(na) == permutedims(denamed(nb), (2, 3, 1))

        a = randn(elt, 3, 4)
        na = nameddims(a, ("i", "j"))
        i = namedoneto(3, "i")
        j = namedoneto(4, "j")
        ai, aj = axes(na)
        for na′ in (
                similar(na, Float32, (j, i)),
                similar(na, Float32, LittleSet((j, i))),
                similar(na, Float32, (aj, ai)),
                similar(na, Float32, LittleSet((aj, ai))),
                similar(a, Float32, (j, i)),
                similar(a, Float32, LittleSet((j, i))),
                similar(a, Float32, (aj, ai)),
                similar(a, Float32, LittleSet((aj, ai))),
            )
            @test eltype(na′) ≡ Float32
            @test all(inds(na′) .== (j, i))
            @test na′ ≠ na
        end

        a = randn(elt, 3, 4)
        na = nameddims(a, ("i", "j"))
        i = namedoneto(3, "i")
        j = namedoneto(4, "j")
        ai, aj = axes(na)
        for na′ in (
                similar(na, (j, i)),
                similar(na, LittleSet((j, i))),
                similar(na, (aj, ai)),
                similar(na, LittleSet((aj, ai))),
                similar(a, (j, i)),
                similar(a, LittleSet((j, i))),
                similar(a, (aj, ai)),
                similar(a, LittleSet((aj, ai))),
            )
            @test eltype(na′) ≡ eltype(na)
            @test all(inds(na′) .== (j, i))
            @test na′ ≠ na
        end

        # getindex syntax
        i = Name("i")
        j = Name("j")
        @test a[i, j] == na
        @test @view(a[i, j]) == na
        @test na[j[1], i[2]] == a[2, 1]
        @test inds(na[j, i]) == (named(1:3, "i"), named(1:4, "j"))
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
        @test denamed(na) == a

        # slicing
        a = randn(elt, 3, 3)
        na = NamedDimsArray(a, ("i", "j"))
        for na′ in (na[named(2:3, "i"), named(2:3, "j")], na["i" => 2:3, "j" => 2:3])
            @test inds(na′) == (named(2:3, "i"), named(2:3, "j"))
            @test denamed(na′) == a[2:3, 2:3]
            @test denamed(na′) isa typeof(a)
        end

        # view slicing
        a = randn(elt, 3, 3)
        na = NamedDimsArray(a, ("i", "j"))
        for na′ in
            (@view(na[named(2:3, "i"), named(2:3, "j")]), @view(na["i" => 2:3, "j" => 2:3]))
            @test inds(na′) == (named(2:3, "i"), named(2:3, "j"))
            @test copy(denamed(na′)) == a[2:3, 2:3]
            @test denamed(na′) ≡ @view(a[2:3, 2:3])
            @test denamed(na′) isa SubArray{elt, 2}
        end

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
        na = nameddims(a, ("i", "j"))
        a′ = denamed(na)
        @test a′ isa Matrix{elt}
        @test a′ == a
        a′ = dename(na, ("j", "i"))
        @test a′ isa Matrix{elt}
        @test a′ == transpose(a)
        a′ = denamed(na, ("j", "i"))
        @test a′ isa PermutedDimsArray{elt}
        @test a′ == transpose(a)
        nb = setinds(na, ("k", "j"))
        @test inds(nb) == (named(1:3, "k"), named(1:4, "j"))
        @test denamed(nb) == a
        nb = replaceinds(na, "i" => "k")
        @test inds(nb) == (named(1:3, "k"), named(1:4, "j"))
        @test denamed(nb) == a
        nb = replaceinds(na, named(1:3, "i") => named(1:3, "k"))
        @test inds(nb) == (named(1:3, "k"), named(1:4, "j"))
        @test denamed(nb) == a
        nb = replaceinds(n -> n == named(1:3, "i") ? named(1:3, "k") : n, na)
        @test inds(nb) == (named(1:3, "k"), named(1:4, "j"))
        @test denamed(nb) == a
        nb = mapinds(n -> n == named(1:3, "i") ? named(1:3, "k") : n, na)
        @test inds(nb) == (named(1:3, "k"), named(1:4, "j"))
        @test denamed(nb) == a
        nb = setinds(na, named(3, "i") => named(3, "k"))
        na[1, 1] = 11
        @test na[1, 1] == 11
        @test Tuple(size(na)) == (named(3, "i"), named(4, "j"))
        @test length(na) == named(12, fusednames("i", "j"))
        @test Tuple(axes(na)) == (named(1:3, "i"), named(1:4, "j"))
        @test randn(named.((3, 4), ("i", "j"))) isa NamedDimsArray
        @test na["i" => 1, "j" => 2] == a[1, 2]
        @test na["j" => 2, "i" => 1] == a[1, 2]
        na["j" => 2, "i" => 1] = 12
        @test na[1, 2] == 12
        @test na[j => 1, i => 2] == a[2, 1]
        na[j => 1, i => 2] = 21
        @test na[2, 1] == 21
        na′ = aligndims(na, ("j", "i"))
        @test denamed(na′) isa Matrix{elt}
        @test a == permutedims(denamed(na′), (2, 1))
        na′ = aligneddims(na, ("j", "i"))
        @test denamed(na′) isa PermutedDimsArray{elt}
        @test a == permutedims(denamed(na′), (2, 1))
        na′ = aligndims(na, (j, i))
        @test denamed(na′) isa Matrix{elt}
        @test a == permutedims(denamed(na′), (2, 1))
        na′ = aligneddims(na, (j, i))
        @test denamed(na′) isa PermutedDimsArray{elt}
        @test a == permutedims(denamed(na′), (2, 1))

        na = nameddims(randn(elt, 2, 3), (:i, :j))
        nb = nameddims(randn(elt, 3, 2), (:j, :i))
        nc = zeros(elt, named.((2, 3), (:i, :j)))
        Is = eachindex(na, nb)
        @test Is isa NamedDimsCartesianIndices{2}
        @test issetequal(Is.indices, (named(1:2, :i), named(1:3, :j)))
        for I in Is
            @test I isa NamedDimsCartesianIndex{2}
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

        # Regression test for proper permutations.
        a = nameddims(randn(elt, 2, 3, 4), (:i, :j, :k))
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
    @testset "begin/end (eltype=$elt)" for elt in elts
        i, j = namedoneto.((2, 3), ("i", "j"))
        a = randn(elt, i, j)
        @test a[begin, begin] == a[1, 1]
        @test a[2, begin] == a[2, 1]
        @test a[begin, 2] == a[1, 2]
        @test a[begin, end] == a[1, 3]
        @test a[end, begin] == a[2, 1]
        @test a[end, end] == a[2, 3]

        @test a[j => begin, i => begin] == a[1, 1]
        @test a[j => 2, i => begin] == a[1, 2]
        @test a[j => begin, i => 2] == a[2, 1]
        @test a[j => begin, i => end] == a[2, 1]
        @test a[j => end, i => begin] == a[1, 3]
        @test a[j => end, i => end] == a[2, 3]

        @test a[j[begin], i[begin]] == a[1, 1]
        @test a[j[2], i[begin]] == a[1, 2]
        @test a[j[begin], i[2]] == a[2, 1]
        @test a[j[begin], i[end]] == a[2, 1]
        @test a[j[end], i[begin]] == a[1, 3]
        @test a[j[end], i[end]] == a[2, 3]
    end
    @testset "Shorthand constructors (eltype=$elt)" for elt in elts
        i, j = named.((2, 2), ("i", "j"))
        value = rand(elt)
        for na in (zeros(elt, i, j), zeros(elt, (i, j)))
            @test eltype(na) ≡ elt
            @test na isa NamedDimsArray
            @test inds(na) == Base.oneto.((i, j))
            @test iszero(na)
        end
        for na in (fill(value, i, j), fill(value, (i, j)))
            @test eltype(na) ≡ elt
            @test na isa NamedDimsArray
            @test inds(na) == Base.oneto.((i, j))
            @test all(isequal(value), na)
        end
        for na in (rand(elt, i, j), rand(elt, (i, j)))
            @test eltype(na) ≡ elt
            @test na isa NamedDimsArray
            @test inds(na) == Base.oneto.((i, j))
            @test !iszero(na)
            @test all(x -> real(x) > 0, na)
        end
        for na in (randn(elt, i, j), randn(elt, (i, j)))
            @test eltype(na) ≡ elt
            @test na isa NamedDimsArray
            @test inds(na) == Base.oneto.((i, j))
            @test !iszero(na)
        end
    end
    @testset "Shorthand constructors (eltype=unspecified)" begin
        i, j = named.((2, 2), ("i", "j"))
        default_elt = Float64
        for na in (zeros(i, j), zeros((i, j)))
            @test eltype(na) ≡ default_elt
            @test na isa NamedDimsArray
            @test inds(na) == Base.oneto.((i, j))
            @test iszero(na)
        end
        for na in (rand(i, j), rand((i, j)))
            @test eltype(na) ≡ default_elt
            @test na isa NamedDimsArray
            @test inds(na) == Base.oneto.((i, j))
            @test !iszero(na)
            @test all(x -> real(x) > 0, na)
        end
        for na in (randn(i, j), randn((i, j)))
            @test eltype(na) ≡ default_elt
            @test na isa NamedDimsArray
            @test inds(na) == Base.oneto.((i, j))
            @test !iszero(na)
        end
    end
    @testset "LittleSet" begin
        # Broadcasting
        s = LittleSet((1, 2))
        @test eltype(s) == Int
        @test s .+ [3, 4] == [4, 6]
        @test s .+ (3, 4) ≡ (4, 6)

        s = LittleSet(("a", "b", "c"))
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
            @test s′ isa LittleSet
            @test Tuple(s′) == ("a", "x", "c")
            @test s′[1] == "a"
            @test s′[2] == "x"
            @test s′[3] == "c"
        end
    end
    false && @testset "show" begin
        a = NamedDimsArray([1 2; 3 4], ("i", "j"))
        @test sprint(show, "text/plain", a) ==
            "named(Base.OneTo(2), \"i\")×named(Base.OneTo(2), \"j\") " *
            "$NamedDimsArray{Int64, 2, Matrix{Int64}, …}:\n" *
            "2×2 Matrix{Int64}:\n 1  2\n 3  4"

        a = NamedDimsArray([1 2; 3 4], ("i", "j"))
        @test sprint(show, a) == "[1 2; 3 4][named(Base.OneTo(2), \"i\"), named(Base.OneTo(2), \"j\")]"
    end

    @testset "operator" begin
        o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
        @test o isa NamedDimsOperator{Float64}
        @test eltype(o) ≡ Float64
        @test issetequal(NDA.domain(o), namedoneto.((2, 2), ("i", "j")))
        @test issetequal(NDA.codomain(o), namedoneto.((2, 2), ("i'", "j'")))

        o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
        õ = similar(o)
        @test õ isa NamedDimsOperator{Float64}
        @test eltype(õ) ≡ Float64
        @test issetequal(NDA.domain(õ), namedoneto.((2, 2), ("i", "j")))
        @test issetequal(NDA.codomain(õ), namedoneto.((2, 2), ("i'", "j'")))

        o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
        õ = similar(o, Float32)
        @test õ isa NamedDimsOperator{Float32}
        @test eltype(õ) ≡ Float32
        @test issetequal(NDA.domain(õ), namedoneto.((2, 2), ("i", "j")))
        @test issetequal(NDA.codomain(õ), namedoneto.((2, 2), ("i'", "j'")))

        o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
        @test o isa NamedDimsOperator
        o² = product(o, o)
        @test issetequal(dimnames(o²), ("i'", "j'", "i", "j"))
        õ = replaceinds(
            state(o), "i" => "i'", "j" => "j'", "i'" => "x", "j'" => "y"
        )
        o²′ = replaceinds(õ * o, "x" => "i'", "y" => "j'")
        @test state(o²) ≈ o²′

        o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
        v = NamedDimsArray(randn(2, 2), ("i", "j"))
        ov = apply(o, v)
        @test issetequal(dimnames(ov), ("i", "j"))
        @test ov ≈ replaceinds(o * v, "i'" => "i", "j'" => "j")
    end

    @testset "@names" begin
        x = @names x
        y, z = @names y z
        a, b, c = @names a[1:2] b[1:2, 1:2] c[2:3, [1, 2]]
        @test x == Name(:x)
        @test y == Name(:y)
        @test z == Name(:z)
        @test size(a) == (2,)
        @test a == [Name(:a_1), Name(:a_2)]
        @test size(b) == (2, 2)
        @test b == [
            Name(:b_1_1) Name(:b_1_2)
            Name(:b_2_1) Name(:b_2_2)
        ]
        @test size(c) == (2, 2)
        @test c == [
            Name(:c_2_1) Name(:c_2_2)
            Name(:c_3_1) Name(:c_3_2)
        ]
    end
end
