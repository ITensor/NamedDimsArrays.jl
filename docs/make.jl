using NamedDimsArrays: NamedDimsArrays
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(
  NamedDimsArrays, :DocTestSetup, :(using NamedDimsArrays); recursive=true
)

include("make_index.jl")

makedocs(;
  modules=[NamedDimsArrays],
  authors="ITensor developers <support@itensor.org> and contributors",
  sitename="NamedDimsArrays.jl",
  format=Documenter.HTML(;
    canonical="https://ITensor.github.io/NamedDimsArrays.jl",
    edit_link="main",
    assets=String[],
  ),
  pages=["Home" => "index.md"],
)

deploydocs(;
  repo="github.com/ITensor/NamedDimsArrays.jl", devbranch="main", push_preview=true
)
