using Adapt: Adapt, adapt

function Adapt.adapt_structure(to, a::AbstractNamedDimsArray)
    return nameddims(adapt(to, denamed(a)), inds(a))
end
