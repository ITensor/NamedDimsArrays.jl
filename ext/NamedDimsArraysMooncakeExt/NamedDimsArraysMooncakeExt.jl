module NamedDimsArraysMooncakeExt

using Mooncake: Mooncake, @zero_derivative, DefaultCtx
using NamedDimsArrays: AbstractNamedDimsArray, AbstractNamedUnitRange,
    blockedperm_nameddims, combine_nameddimsconstructors, dimnames, axes_setdiff,
    name, nameddimsconstructorof, randname, to_axes
using TensorAlgebra: blockedperm

Mooncake.tangent_type(::Type{<:AbstractNamedUnitRange}) = Mooncake.NoTangent

@zero_derivative DefaultCtx Tuple{typeof(blockedperm), AbstractNamedDimsArray, Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(blockedperm_nameddims), Any, Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(combine_nameddimsconstructors), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(dimnames), Any}
@zero_derivative DefaultCtx Tuple{typeof(dimnames), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(axes_setdiff), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(name), Any}
@zero_derivative DefaultCtx Tuple{typeof(nameddimsconstructorof), Any}
@zero_derivative DefaultCtx Tuple{typeof(randname), Any}
@zero_derivative DefaultCtx Tuple{typeof(randname), Any, Any}
@zero_derivative DefaultCtx Tuple{typeof(to_axes), Any, Any}

using Mooncake: Tangent
using NamedDimsArrays: AbstractNamedDimsArray, NamedDimsArray, denamed
function Base.copyto!(dest::NamedDimsArray, src::Tangent)
    # TODO: Account for the named `axes` of the Tangent? In other words, is the tangent data
    # aligned with the `dest` data?
    copyto!(denamed(dest), src.fields.parent)
    return dest
end

end
