using TypeParameterAccessors: TypeParameterAccessors, parenttype

# inds should be a named slice.
struct NamedDimsArray{T, N, Denamed <: AbstractArray{T, N}, DimNames <: Tuple{Vararg{Any, N}}} <:
    AbstractNamedDimsArray{T, N}
    denamed::Denamed
    dimnames::DimNames
    function NamedDimsArray{T, N, Denamed, DimNames}(
            denamed::AbstractArray{<:Any, N}, dims::Tuple{Vararg{Any, N}}
        ) where {T, N, Denamed <: AbstractArray{T, N}, DimNames <: Tuple{Vararg{Any, N}}}
        dimnames = to_dimnames(denamed, dims)
        return new{T, N, Denamed, DimNames}(denamed, dimnames)
    end
end
function NamedDimsArray(
        denamed::Denamed, dims::Tuple{Vararg{Any, N}}
    ) where {T, N, Denamed <: AbstractArray{T, N}}
    # This checks the shapes of the inputs.
    dimnames = to_dimnames(denamed, dims)
    return NamedDimsArray{T, N, Denamed, typeof(dimnames)}(denamed, dimnames)
end

const NamedDimsVector{T, Denamed <: AbstractVector{T}, Inds <: Tuple{Any}} = NamedDimsArray{
    T, 1, Denamed, Inds,
}
const NamedDimsMatrix{T, Denamed <: AbstractMatrix{T}, Inds <: Tuple{Any, Any}} = NamedDimsArray{
    T, 2, Denamed, Inds,
}

function NamedDimsArray(denamed::AbstractArray, dims)
    ndims(denamed) == length(dims) || throw(ArgumentError("Number of named dims must match ndims."))
    return NamedDimsArray(denamed, Tuple{Vararg{Any, ndims(denamed)}}(dims))
end

# TODO: Delete this, and just wrap the input naively.
function NamedDimsArray(a::AbstractNamedDimsArray, inds)
    return error("Already named.")
end

function NamedDimsArray(a::AbstractNamedDimsArray)
    return NamedDimsArray(denamed(a), inds(a))
end

# Minimal interface.
dimnames(a::NamedDimsArray) = LittleSet(a.dimnames)
denamed(a::NamedDimsArray) = a.denamed
Base.parent(a::NamedDimsArray) = denamed(a)

function TypeParameterAccessors.position(
        ::Type{<:AbstractNamedDimsArray}, ::typeof(parenttype)
    )
    return TypeParameterAccessors.Position(3)
end
