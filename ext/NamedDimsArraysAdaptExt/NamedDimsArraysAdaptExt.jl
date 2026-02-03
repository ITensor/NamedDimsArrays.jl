module NamedDimsArraysAdaptExt

using Adapt: Adapt, adapt
using NamedDimsArrays: AbstractNamedDimsArray, denamed, dimnames, nameddims

function Adapt.adapt_structure(to, a::AbstractNamedDimsArray)
    return nameddims(adapt(to, denamed(a)), dimnames(a))
end

end
