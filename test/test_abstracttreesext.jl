using AbstractTrees: printnode
using NamedDimsArrays: nameddims
using Test: @test, @testset

@testset "AbstractTreesExt" begin
    a = randn(3, 4)
    na = nameddims(a, ("i", "j"))
    @test sprint(printnode, na) == """("i", "j")"""
end
