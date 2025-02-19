module NamedDimsArrays

export NamedDimsArray, aligndims, named, nameddimsarray
using Compat: @compat
#! format: off
@compat public to_nameddimsindices
#! format: on

include("isnamed.jl")
include("randname.jl")
include("abstractnamedinteger.jl")
include("namedinteger.jl")
include("abstractnamedarray.jl")
include("namedarray.jl")
include("abstractnamedunitrange.jl")
include("namedunitrange.jl")
include("abstractnameddimsarray.jl")
include("adapt.jl")
include("tensoralgebra.jl")
include("nameddimsarray.jl")

end
