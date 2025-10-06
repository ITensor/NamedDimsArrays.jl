using Adapt: adapt
using DiagonalArrays: DiagonalArray
using JLArrays: JLArray
using NamedDimsArrays: dename, nameddims, nameddimsindices
using SparseArraysBase: sparsezeros, dense
using Test: @test, @testset

@testset "SparseArraysBaseExt (eltype=$elt, arraytype=$arrayt)" for elt in (Float64, ComplexF64),
        arrayt in (Array, JLArray)

    dev = adapt(arrayt)

    @testset "SparseArrayDOK" begin
        s = sparsezeros(elt, 3, 4)
        a = nameddims(s, (:a, :b))
        b = dense(a)
        @test dename(b) == dense(dename(a))
        @test dename(b) isa Array{elt, 2}
        @test nameddimsindices(b) == nameddimsindices(a)
    end

    @testset "DiagonalArrays" begin
        s = dev(DiagonalArray(randn(elt, 3), (3, 3)))
        a = nameddims(s, (:a, :b))
        b = dense(a)
        @test dename(b) == dense(dename(a))
        @test dename(b) isa arrayt{elt, 2}
        @test nameddimsindices(b) == nameddimsindices(a)
    end
end
