module NamedDimsArraysAbstractTreesExt

using AbstractTrees: AbstractTrees
using NamedDimsArrays: AbstractNamedDimsArray, dimnames

# Only print the dimension names when printing with `AbstractTrees.print_tree`.
function AbstractTrees.printnode(io::IO, a::AbstractNamedDimsArray)
    dimnames_a = "{" * join(map(s -> "\"$s\"", dimnames(a)), ", ") * "}"
    print(io, dimnames_a)
    return nothing
end

end
