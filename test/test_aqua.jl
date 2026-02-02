using NamedDimsArrays: NamedDimsArrays
using Aqua: Aqua
using Test: @testset

@testset "Code quality (Aqua.jl)" begin
    # TODO: Fix and re-enable ambiguity checks.
    # For some reason, `persistent_tasks` checks fail in Julia v1.10, because
    # precompilation has trouble with `NamedDimsArraysAdaptExt` and
    # `NamedDimsArraysBlockArraysExt`:
    # ```
    # ┌ Warning: Circular dependency detected.
│   # Precompilation will be skipped for dependencies in this cycle:
│   #  ┌ NamedDimsArrays → NamedDimsArraysAdaptExt
│   #  └─ NamedDimsArrays → NamedDimsArraysBlockArraysExt
    # ```
    # TODO: Remove the `persistent_tasks` condition when Julia v1.10 is no longer supported.
    Aqua.test_all(
        NamedDimsArrays;
        ambiguities = false,
        persistent_tasks = VERSION ≥ v"1.12-",
    )
end
