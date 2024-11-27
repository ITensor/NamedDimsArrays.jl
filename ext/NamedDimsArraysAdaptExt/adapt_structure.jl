using Adapt: Adapt, adapt
using ..NamedDimsArrays: AbstractNamedDimsArray, dimnames, named, unname

function Adapt.adapt_structure(to, na::AbstractNamedDimsArray)
  return named(adapt(to, unname(na)), dimnames(na))
end
