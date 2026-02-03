using LinearAlgebra: mul!
import Mooncake
using NamedDimsArrays: AbstractNamedUnitRange, Name, NamedDimsArray, NamedUnitRange,
    blockedperm_nameddims, combine_nameddimsconstructors, dimnames, dimnames_setdiff, inds,
    name, nameddimsconstructorof, randname, to_inds
import Random
using TensorAlgebra: blockedperm
using Test: @test, @testset

@testset "MooncakeExt" begin
    elt = Float64
    mode = Mooncake.ReverseMode
    rng = Random.default_rng()
    is_primitive = false
    atol = eps(real(elt))^(3 / 4)
    rtol = eps(real(elt))^(3 / 4)
    @testset "zero derivatives" begin
        @test Mooncake.tangent_type(AbstractNamedUnitRange) ≡ Mooncake.NoTangent
        @test Mooncake.tangent_type(NamedUnitRange) ≡ Mooncake.NoTangent

        i, j, k = Name.((:i, :j, :k))
        dest = randn(elt, (2, 2))[i, k]
        a1 = randn(elt, (2, 2))[i, j]
        a2 = randn(elt, (2, 2))[j, k]

        Mooncake.TestUtils.test_rule(rng, blockedperm, a1, (i,), (j,); mode, is_primitive)
        Mooncake.TestUtils.test_rule(
            rng, blockedperm_nameddims, a1, (i,), (j,); mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng, combine_nameddimsconstructors, NamedDimsArray, NamedDimsArray;
            mode, is_primitive,
        )
        Mooncake.TestUtils.test_rule(rng, dimnames, a1; mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, dimnames, a1, 1; mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, inds, a1; mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, inds, a1, 1; mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, dimnames_setdiff, (i, j), (j, k); mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, name, i; mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, nameddimsconstructorof, a1; mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, randname, i; mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, randname, rng, i; mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, to_inds, a1, (i, j); mode, is_primitive)
    end
    @testset "contract" begin
        i, j, k = Name.((:i, :j, :k))
        dest = randn(elt, (2, 2))[i, k]
        a1 = randn(elt, (2, 2))[i, j]
        a2 = randn(elt, (2, 2))[j, k]
        Mooncake.TestUtils.test_rule(rng, *, a1, a2; atol, rtol, mode, is_primitive)
        Mooncake.TestUtils.test_rule(
            rng, mul!, dest, a1, a2; atol, rtol, mode, is_primitive
        )
    end
end
