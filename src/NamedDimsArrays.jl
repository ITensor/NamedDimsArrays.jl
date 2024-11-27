module NamedDimsArrays

include("traits.jl")
include("name.jl")
include("randname.jl")
include("abstractnamedint.jl")
include("abstractnamedunitrange.jl")
include("abstractnameddimsarray.jl")
include("abstractnameddimsmatrix.jl")
include("abstractnameddimsvector.jl")
include("namedint.jl")
include("namedunitrange.jl")
include("nameddimsarray.jl")
include("constructors.jl")
include("similar.jl")
include("permutedims.jl")
include("promote_shape.jl")
include("map.jl")
include("broadcast_shape.jl")
include("broadcast.jl")

# Extensions.
# TODO: Turn these into actual package extensions.
include("../ext/NamedDimsArraysAdaptExt/NamedDimsArraysAdaptExt.jl")
include("../ext/NamedDimsArraysSparseArraysBaseExt/NamedDimsArraysSparseArraysBaseExt.jl")
include("../ext/NamedDimsArraysTensorAlgebraExt/NamedDimsArraysTensorAlgebraExt.jl")

end
