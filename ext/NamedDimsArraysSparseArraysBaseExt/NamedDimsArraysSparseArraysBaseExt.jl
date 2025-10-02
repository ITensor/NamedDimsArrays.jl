module NamedDimsArraysSparseArraysBaseExt

using NamedDimsArrays:
    AbstractNamedDimsArray, constructorof_nameddimsarray, dename, nameddimsindices
using SparseArraysBase: SparseArraysBase, dense

function SparseArraysBase.dense(a::AbstractNamedDimsArray)
    return constructorof_nameddimsarray(typeof(a))(dense(dename(a)), nameddimsindices(a))
end

end
