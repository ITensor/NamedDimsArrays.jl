module NamedDimsArrays

export NamedDimsArray, aligndims, named, nameddims, operator
using Compat: @compat
@compat public to_nameddimsindices
@compat public @names

include("naiveorderedset.jl")
include("isnamed.jl")
include("randname.jl")
include("abstractnamedinteger.jl")
include("namedinteger.jl")
include("abstractnamedarray.jl")
include("namedarray.jl")
include("abstractnamedunitrange.jl")
include("namedunitrange.jl")
include("abstractnameddims.jl")
include("adapt.jl")
include("tensoralgebra.jl")
include("nameddims.jl")
include("nameddimsoperator.jl")

end
