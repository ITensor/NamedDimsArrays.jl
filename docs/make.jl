using Documenter: Documenter, DocMeta, deploydocs, makedocs
using ITensorFormatter: ITensorFormatter
using NamedDimsArrays: NamedDimsArrays

DocMeta.setdocmeta!(
    NamedDimsArrays, :DocTestSetup, :(using NamedDimsArrays); recursive = true
)

ITensorFormatter.make_index!(pkgdir(NamedDimsArrays))

makedocs(;
    modules = [NamedDimsArrays],
    authors = "ITensor developers <support@itensor.org> and contributors",
    sitename = "NamedDimsArrays.jl",
    format = Documenter.HTML(;
        canonical = "https://itensor.github.io/NamedDimsArrays.jl",
        edit_link = "main",
        assets = ["assets/favicon.ico", "assets/extras.css"]
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"]
)

deploydocs(;
    repo = "github.com/ITensor/NamedDimsArrays.jl", devbranch = "main", push_preview = true
)
