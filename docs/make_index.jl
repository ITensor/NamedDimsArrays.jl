using Literate: Literate
using NamedDimsArrays: NamedDimsArrays

Literate.markdown(
  joinpath(pkgdir(NamedDimsArrays), "examples", "README.jl"),
  joinpath(pkgdir(NamedDimsArrays), "docs", "src");
  flavor=Literate.DocumenterFlavor(),
  name="index",
)
