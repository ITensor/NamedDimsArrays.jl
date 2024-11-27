# # NamedDimsArrays.jl
# 
# [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ITensor.github.io/NamedDimsArrays.jl/stable/)
# [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ITensor.github.io/NamedDimsArrays.jl/dev/)
# [![Build Status](https://github.com/ITensor/NamedDimsArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ITensor/NamedDimsArrays.jl/actions/workflows/CI.yml?query=branch%3Amain)
# [![Coverage](https://codecov.io/gh/ITensor/NamedDimsArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ITensor/NamedDimsArrays.jl)
# [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
# [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# ## Installation instructions

#=
```julia
julia> using Pkg: Pkg

julia> Pkg.add("https://github.com/ITensor/NamedDimsArrays.jl")
```
=#

# ## Examples

using NamedDimsArrays: align, dimnames, named, unname
using TensorAlgebra: contract

## Named dimensions
i = named(2, "i")
j = named(2, "j")
k = named(2, "k")

## Arrays with named dimensions
na1 = randn(i, j)
na2 = randn(j, k)

@show dimnames(na1) == ("i", "j")

## Indexing
@show na1[j => 2, i => 1] == na1[1, 2]

## Tensor contraction
na_dest = contract(na1, na2)

@show issetequal(dimnames(na_dest), ("i", "k"))
## `unname` removes the names and returns an `Array`
@show unname(na_dest, (i, k)) ≈ unname(na1) * unname(na2)

## Permute dimensions (like `ITensors.permute`)
na1 = align(na1, (j, i))
@show na1[i => 1, j => 2] == na1[2, 1]
