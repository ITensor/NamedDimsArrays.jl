module NamedDimsArrays

export NamedDimsArray, aligndims, named, nameddims, operator
using Compat: @compat
@compat public to_inds
@compat public @names

include("littleset.jl")
include("isnamed.jl")
include("randname.jl")
include("abstractnamedinteger.jl")
include("namedinteger.jl")
include("abstractnamedarray.jl")
include("namedarray.jl")
include("abstractnamedunitrange.jl")
include("namedunitrange.jl")
include("abstractnameddimsarray.jl")
include("lazynameddimsarray.jl")
include("broadcast.jl")
include("adapt.jl")
include("tensoralgebra.jl")
include("nameddimsarray.jl")
include("nameddimsoperator.jl")

end
