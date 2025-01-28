using BlockArrays: Block
using BlockSparseArrays: BlockSparseArray
using NamedDimsArrays: dename, nameddimsarray, nameddimsindices
using Test: @test, @testset

@testset "NamedDimsArraysBlockArraysExt" begin
  elt = Float64
  a = BlockSparseArray{elt}([2, 3], [2, 3])
  a[Block(2, 1)] = randn(elt, 3, 2)
  a[Block(1, 2)] = randn(elt, 2, 3)
  n = nameddimsarray(a, ("i", "j"))
  i, j = nameddimsindices(n)
  @test dename(n[i[Block(2)], j[Block(1)]]) == a[Block(2, 1)]
  @test dename(n[Block(2), Block(1)]) == a[Block(2, 1)]
  @test dename(n[Block(2, 1)]) == a[Block(2, 1)]
  @test dename(n[i[Block(2)], j[Block.(1:2)]]) == a[Block(2), Block.(1:2)]
  @test dename(n[Block(2), Block.(1:2)]) == a[Block(2), Block.(1:2)]
  @test dename(n[i[Block.(1:2)], j[Block(1)]]) == a[Block.(1:2), Block(1)]
  @test dename(n[Block.(1:2), Block(1)]) == a[Block.(1:2), Block(1)]
  @test dename(n[Block.(1:2), Block.(1:2)]) == a[Block.(1:2), Block.(1:2)]
end
