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

using NamedDimsArrays: aligndims, dimnames, named, nameddimsindices, namedoneto, unname
using TensorAlgebra: contract
using Test: @test

## Named dimensions
i = namedoneto(2, "i")
j = namedoneto(2, "j")
k = namedoneto(2, "k")

## Arrays with named dimensions
a1 = randn(i, j)
a2 = randn(j, k)

@test dimnames(a1) == ("i", "j")
@test nameddimsindices(a1) == (i, j)
@test axes(a1) == (named(1:2, i), named(1:2, j))
@test size(a1) == (named(2, i), named(2, j))

## Indexing
@test a1[j => 2, i => 1] == a1[1, 2]
@test a1[j[2], i[1]] == a1[1, 2]

## Tensor contraction
a_dest = contract(a1, a2)

@test issetequal(nameddimsindices(a_dest), (i, k))
## `unname` removes the names and returns an `Array`
@test unname(a_dest, (i, k)) ≈ unname(a1, (i, j)) * unname(a2, (j, k))

## Permute dimensions (like `ITensors.permute`)
a1′ = aligndims(a1, (j, i))
@test a1′[i => 1, j => 2] == a1[i => 1, j => 2]
@test a1′[i[1], j[2]] == a1[i[1], j[2]]

## Contiguous slicing
b1 = a1[i => 1:2, j => 1:1]
@test b1 == a1[i[1:2], j[1:1]]

b2 = a2[j => 1:1, k => 1:2]
@test b2 == a2[j[1:1], k[1:2]]

@test nameddimsindices(b1) == (i[1:2], j[1:1])
@test nameddimsindices(b2) == (j[1:1], k[1:2])

b_dest = contract(b1, b2)

@test issetequal(nameddimsindices(b_dest), (i, k))

## Non-contiguous slicing
c1 = a1[i[[2, 1]], j[[2, 1]]]
@test nameddimsindices(c1) == (i[[2, 1]], j[[2, 1]])
@test unname(c1, (i[[2, 1]], j[[2, 1]])) == unname(a1, (i, j))[[2, 1], [2, 1]]
@test c1[i[2], j[1]] == a1[i[2], j[1]]
@test c1[2, 1] == a1[1, 2]

a1[i[[2, 1]], j[[2, 1]]] = [22 21; 12 11]
@test a1[i[1], j[1]] == 11

x = randn(i[1:2], j[2:2])
a1[i[1:2], j[2:2]] = x
@test a1[i[1], j[2]] == x[i[1], j[2]]
