module NamedDimsArraysAbstractTreesExt

using AbstractTrees: AbstractTrees
using NamedDimsArrays: AbstractNamedDimsArray, dimnames

# Only print the dimension names when printing with `AbstractTrees.print_tree`.
function AbstractTrees.printnode(io::IO, a::AbstractNamedDimsArray)
    show(IOContext(io, :compact => true, :limit => true), dimnames(a))
    return nothing
end

end
