using TypeParameterAccessors:
    TypeParameterAccessors, Position, get_type_parameters, parenttype

# TODO: Check `allunique(dimnames)`?
struct NamedDimsArray{
        T,
        N,
        Denamed <: AbstractArray{T, N},
        DimNames <: Tuple{Vararg{Any, N}},
    } <:
    AbstractNamedDimsArray{T, N}
    denamed::Denamed
    dimnames::DimNames
end

const NamedDimsVector{T, Denamed <: AbstractVector{T}, Inds <: Tuple{Any}} = NamedDimsArray{
    T, 1, Denamed, Inds,
}
const NamedDimsMatrix{T, Denamed <: AbstractMatrix{T}, Inds <: Tuple{Any, Any}} =
    NamedDimsArray{
    T, 2, Denamed, Inds,
}

# TODO: Check `allunique(dimnames)`?
function NamedDimsArray(denamed::AbstractArray, dims)
    ndims(denamed) == length(dims) ||
        throw(ArgumentError("Number of named dims must match ndims."))
    return NamedDimsArray(denamed, Tuple{Vararg{Any, ndims(denamed)}}(dims))
end
NamedDimsArray(a::AbstractNamedDimsArray, inds) = throw(ArgumentError("Already named."))
NamedDimsArray(a::AbstractNamedDimsArray) = NamedDimsArray(denamed(a), dimnames(a))

# Minimal interface.
dimnames(a::NamedDimsArray) = LittleSet(a.dimnames)
denamed(a::NamedDimsArray) = a.denamed
Base.parent(a::NamedDimsArray) = denamed(a)

denamedtype(T::Type{<:NamedDimsArray}) = get_type_parameters(T, Position(3))
nametype(T::Type{<:NamedDimsArray}) = eltype(get_type_parameters(T, Position(4)))

function TypeParameterAccessors.position(
        ::Type{<:AbstractNamedDimsArray}, ::typeof(parenttype)
    )
    return Position(3)
end
