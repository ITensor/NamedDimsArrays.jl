using BlockArrays: Block, BlockArray
using NamedDimsArrays: dename, denamed, inds, nameddims
using Test: @test, @testset

@testset "NamedDimsArraysBlockArraysExt" begin
    elt = Float64

    a = BlockArray{elt}(undef, [2, 3], [2, 3])
    a[Block(1, 1)] = randn(elt, 2, 2)
    a[Block(2, 1)] = randn(elt, 3, 2)
    a[Block(1, 2)] = randn(elt, 2, 3)
    a[Block(2, 2)] = randn(elt, 3, 3)
    n = nameddims(a, ("i", "j"))
    i, j = inds(n)
    @test denamed(n[i[Block(2)], j[Block(1)]]) == a[Block(2, 1)]
    @test denamed(n[Block(2), Block(1)]) == a[Block(2, 1)]
    @test denamed(n[Block(2, 1)]) == a[Block(2, 1)]
    @test denamed(n[i[Block(2)], j[Block.(1:2)]]) == a[Block(2), Block.(1:2)]
    @test denamed(n[Block(2), Block.(1:2)]) == a[Block(2), Block.(1:2)]
    @test denamed(n[i[Block.(1:2)], j[Block(1)]]) == a[Block.(1:2), Block(1)]
    @test denamed(n[Block.(1:2), Block(1)]) == a[Block.(1:2), Block(1)]
    @test denamed(n[Block.(1:2), Block.(1:2)]) == a[Block.(1:2), Block.(1:2)]

    a = BlockArray{elt}(undef, [2, 3], [2, 3])
    a[Block(1, 1)] = randn(elt, 2, 2)
    a[Block(2, 1)] = randn(elt, 3, 2)
    a[Block(1, 2)] = randn(elt, 2, 3)
    a[Block(2, 2)] = randn(elt, 3, 3)
    b = BlockArray{elt}(undef, [2, 3], [2, 3])
    b[Block(1, 1)] = randn(elt, 2, 2)
    b[Block(2, 1)] = randn(elt, 3, 2)
    b[Block(1, 2)] = randn(elt, 2, 3)
    b[Block(2, 2)] = randn(elt, 3, 3)
    na = nameddims(a, ("i", "j"))
    nb = nameddims(b, ("j", "i"))
    nc = na .+ 2 .* nb
    c = a + 2 * permutedims(b, (2, 1))
    @test dename(nc, ("i", "j")) ≈ c
end
