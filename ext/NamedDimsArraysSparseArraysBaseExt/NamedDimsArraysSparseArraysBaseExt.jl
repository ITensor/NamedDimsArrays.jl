module NamedDimsArraysSparseArraysBaseExt

using NamedDimsArrays:
    AbstractNamedDimsArray, constructorof_nameddims, dename, inds
using SparseArraysBase: SparseArraysBase, dense

function SparseArraysBase.dense(a::AbstractNamedDimsArray)
    return constructorof_nameddims(typeof(a))(dense(dename(a)), inds(a))
end

end
