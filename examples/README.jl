# # NamedDimsArrays.jl
# 
# [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/NamedDimsArrays.jl/stable/)
# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/NamedDimsArrays.jl/dev/)
# [![Build Status](https://github.com/ITensor/NamedDimsArrays.jl/actions/workflows/Tests.yml/badge.svg?branch=main)](https://github.com/ITensor/NamedDimsArrays.jl/actions/workflows/Tests.yml?query=branch%3Amain)
# [![Coverage](https://codecov.io/gh/ITensor/NamedDimsArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/NamedDimsArrays.jl)
# [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
# [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# ## Installation instructions

# This package resides in the `ITensor/ITensorRegistry` local registry.
# In order to install, simply add that registry through your package manager.
# This step is only required once.
#=
```julia
julia> using Pkg: Pkg

julia> Pkg.Registry.add(url="https://github.com/ITensor/ITensorRegistry")
```
=#
# or:
#=
```julia
julia> Pkg.Registry.add(url="git@github.com:ITensor/ITensorRegistry.git")
```
=#
# if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.

# Then, the package can be added as usual through the package manager:

#=
```julia
julia> Pkg.add("NamedDimsArrays")
```
=#

# ## Examples

using NamedDimsArrays: aligndims, unname, dimnames, named
using TensorAlgebra: contract
using Test: @test

## Named dimensions
i = named(2, "i")
j = named(2, "j")
k = named(2, "k")

## Arrays with named dimensions
a1 = randn(i, j)
a2 = randn(j, k)

@test dimnames(a1) == ("i", "j")
@test axes(a1) == (named(1:2, "i"), named(1:2, "j"))
@test size(a1) == (named(2, "i"), named(2, "j"))

## Indexing
@test a1[j => 2, i => 1] == a1[1, 2]

## Tensor contraction
a_dest = contract(a1, a2)

@test issetequal(dimnames(a_dest), ("i", "k"))
## `unname` removes the names and returns an `Array`
@test unname(a_dest, (i, k)) ≈ unname(a1, (i, j)) * unname(a2, (j, k))

## Permute dimensions (like `ITensors.permute`)
a1′ = aligndims(a1, (j, i))
@test a1′[i => 1, j => 2] == a1[i => 1, j => 2]
