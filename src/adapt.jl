using Adapt: Adapt, adapt

function Adapt.adapt_structure(to, a::AbstractNamedDimsArray)
  return nameddims(adapt(to, dename(a)), nameddimsindices(a))
end
