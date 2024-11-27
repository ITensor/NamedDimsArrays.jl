using Literate: Literate
using NamedDimsArrays: NamedDimsArrays

Literate.markdown(
  joinpath(pkgdir(NamedDimsArrays), "examples", "README.jl"),
  joinpath(pkgdir(NamedDimsArrays));
  flavor=Literate.CommonMarkFlavor(),
  name="README",
)
