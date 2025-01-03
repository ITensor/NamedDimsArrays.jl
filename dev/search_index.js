var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"EditURL = \"../../examples/README.jl\"","category":"page"},{"location":"#NamedDimsArrays.jl","page":"Home","title":"NamedDimsArrays.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Stable) (Image: Dev) (Image: Build Status) (Image: Coverage) (Image: Code Style: Blue) (Image: Aqua)","category":"page"},{"location":"#Installation-instructions","page":"Home","title":"Installation instructions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package resides in the ITensor/ITensorRegistry local registry. In order to install, simply add that registry through your package manager. This step is only required once.","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using Pkg: Pkg\n\njulia> Pkg.Registry.add(url=\"https://github.com/ITensor/ITensorRegistry\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"or:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> Pkg.Registry.add(url=\"git@github.com:ITensor/ITensorRegistry.git\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"if you want to use SSH credentials, which can make it so you don't have to enter your Github ursername and password when registering packages.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Then, the package can be added as usual through the package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> Pkg.add(\"NamedDimsArrays\")","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using NamedDimsArrays: aligndims, dimnames, named, nameddimsindices, namedoneto, unname\nusing TensorAlgebra: contract\nusing Test: @test\n\n# Named dimensions\ni = namedoneto(2, \"i\")\nj = namedoneto(2, \"j\")\nk = namedoneto(2, \"k\")\n\n# Arrays with named dimensions\na1 = randn(i, j)\na2 = randn(j, k)\n\n@test dimnames(a1) == (\"i\", \"j\")\n@test nameddimsindices(a1) == (i, j)\n@test axes(a1) == (named(1:2, i), named(1:2, j))\n@test size(a1) == (named(2, i), named(2, j))\n\n# Indexing\n@test a1[j => 2, i => 1] == a1[1, 2]\n@test a1[j[2], i[1]] == a1[1, 2]\n\n# Tensor contraction\na_dest = contract(a1, a2)\n\n@test issetequal(nameddimsindices(a_dest), (i, k))\n# `unname` removes the names and returns an `Array`\n@test unname(a_dest, (i, k)) ≈ unname(a1, (i, j)) * unname(a2, (j, k))\n\n# Permute dimensions (like `ITensors.permute`)\na1′ = aligndims(a1, (j, i))\n@test a1′[i => 1, j => 2] == a1[i => 1, j => 2]\n@test a1′[i[1], j[2]] == a1[i[1], j[2]]\n\n# Contiguous slicing\nb1 = a1[i => 1:2, j => 1:1]\n@test b1 == a1[i[1:2], j[1:1]]\n\nb2 = a2[j => 1:1, k => 1:2]\n@test b2 == a2[j[1:1], k[1:2]]\n\n@test nameddimsindices(b1) == (i[1:2], j[1:1])\n@test nameddimsindices(b2) == (j[1:1], k[1:2])\n\nb_dest = contract(b1, b2)\n\n@test issetequal(nameddimsindices(b_dest), (i, k))\n\n# Non-contiguous slicing\nc1 = a1[i[[2, 1]], j[[2, 1]]]\n@test nameddimsindices(c1) == (i[[2, 1]], j[[2, 1]])\n@test unname(c1, (i[[2, 1]], j[[2, 1]])) == unname(a1, (i, j))[[2, 1], [2, 1]]\n@test c1[i[2], j[1]] == a1[i[2], j[1]]\n@test c1[2, 1] == a1[1, 2]\n\na1[i[[2, 1]], j[[2, 1]]] = [22 21; 12 11]\n@test a1[i[1], j[1]] == 11\n\nx = randn(i[1:2], j[2:2])\na1[i[1:2], j[2:2]] = x\n@test a1[i[1], j[2]] == x[i[1], j[2]]","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"This page was generated using Literate.jl.","category":"page"}]
}
