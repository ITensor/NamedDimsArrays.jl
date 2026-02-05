using NamedDimsArrays: NamedDimsArrays as NDA, NamedDimsArray, NamedDimsOperator, apply,
    dimnames, namedoneto, operator, product, replacedimnames, state
using Test: @test, @testset

@testset "operator" begin
    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    @test o isa NamedDimsOperator{Float64}
    @test eltype(o) ≡ Float64
    @test issetequal(NDA.codomainnames(o), ("i'", "j'"))
    @test issetequal(NDA.domainnames(o), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    õ = similar(o)
    @test õ isa NamedDimsOperator{Float64}
    @test eltype(õ) ≡ Float64
    @test issetequal(NDA.codomainnames(õ), ("i'", "j'"))
    @test issetequal(NDA.domainnames(õ), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    õ = similar(o, Float32)
    @test õ isa NamedDimsOperator{Float32}
    @test eltype(õ) ≡ Float32
    @test issetequal(NDA.codomainnames(õ), ("i'", "j'"))
    @test issetequal(NDA.domainnames(õ), ("i", "j"))

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    @test o isa NamedDimsOperator
    o² = product(o, o)
    @test issetequal(dimnames(o²), ("i'", "j'", "i", "j"))
    õ = replacedimnames(
        state(o), "i" => "i'", "j" => "j'", "i'" => "x", "j'" => "y"
    )
    o²′ = replacedimnames(õ * o, "x" => "i'", "y" => "j'")
    @test state(o²) ≈ o²′

    o = operator(randn(2, 2, 2, 2), ("i'", "j'"), ("i", "j"))
    v = NamedDimsArray(randn(2, 2), ("i", "j"))
    ov = apply(o, v)
    @test issetequal(dimnames(ov), ("i", "j"))
    @test ov ≈ replacedimnames(o * v, "i'" => "i", "j'" => "j")
end
