using Adapt: Adapt, adapt

function Adapt.adapt_structure(to, a::AbstractNamedDimsArray)
  return nameddimsarray(adapt(to, dename(a)), nameddimsindices(a))
end
