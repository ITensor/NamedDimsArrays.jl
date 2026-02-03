module NamedDimsArraysMooncakeExt

using Mooncake: Mooncake, @zero_derivative, DefaultCtx
using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedUnitRange,
    blockedperm_nameddims, combine_nameddimsconstructors, dimnames, dimnames_setdiff, inds,
    name, nameddimsconstructorof, randname, to_inds
using TensorAlgebra: blockedperm

Mooncake.tangent_type(::Type{<:AbstractNamedUnitRange}) = Mooncake.NoTangent

@zero_derivative DefaultCtx Tuple{typeof(blockedperm), AbstractNamedDimsArray, Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(blockedperm_nameddims), Any, Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(combine_nameddimsconstructors), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(dimnames), Any}
@zero_derivative DefaultCtx Tuple{typeof(dimnames), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(dimnames_setdiff), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(inds), Any}
@zero_derivative DefaultCtx Tuple{typeof(inds), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(name), Any}
@zero_derivative DefaultCtx Tuple{typeof(nameddimsconstructorof), Any}
@zero_derivative DefaultCtx Tuple{typeof(randname), Any}
@zero_derivative DefaultCtx Tuple{typeof(randname), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(to_inds), Any, Any}

using Mooncake: Tangent
using NamedDimsArrays: AbstractNamedDimsArray, NamedDimsArray, denamed
function Base.copyto!(dest::NamedDimsArray, src::Tangent)
    # TODO: Account for the `inds` of the Tangent? In other words, is the tangent data
    # aligned with the `dest` data?
    copyto!(denamed(dest), src.fields.parent)
    return dest
end

end
