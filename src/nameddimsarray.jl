using TypeParameterAccessors: TypeParameterAccessors, parenttype

# axes should be a named slice.
struct NamedDimsArray{
        T, N, Denamed <: AbstractArray{T, N},
        DimNames <: Tuple{Vararg{Any, N}}, Inds <: Tuple{Vararg{Any, N}}
    } <: AbstractNamedDimsArray{T, N}
    denamed::Denamed
    dimnames::DimNames
    inds::Inds
    ## function NamedDimsArray{T, N, Denamed, DimNames, Inds}(
    ##         parent::AbstractArray{<:Any, N},
    ##         dimnames::Tuple{Vararg{Any, N}},
    ##         inds::Tuple{Vararg{Any, N}},
    ##     ) where {
    ##         T, N, Denamed <: AbstractArray{T, N},
    ##         DimNames <: Tuple{Vararg{Any, N}}, Inds <: Tuple{Vararg{Any, N}},
    ##     }
    ##     return new{T, N, Denamed, DimNames, Inds}(parent, dimnames, inds)
    ## end
end

function NamedDimsArray(
        parent::Denamed, dims::Tuple{Vararg{Any, N}}
    ) where {T, N, Denamed <: AbstractArray{T, N}}
    # This checks the shapes of the inputs.
    ax = to_axes(parent, dims)
    return NamedDimsArray(parent, keys(ax), values(ax))
end

function NamedDimsArray(parent::AbstractArray{<:Any, N}, dimnames, inds) where {N}
    return NamedDimsArray(parent, Tuple{Vararg{Any, N}}(dimnames), Tuple{Vararg{Any, N}}(inds))
end

const NamedDimsVector{T, Denamed <: AbstractVector{T}, DimNames <: Tuple{Any}, Inds <: Tuple{Any}} = NamedDimsArray{
    T, 1, Denamed, DimNames, Inds
}
const NamedDimsMatrix{T, Denamed <: AbstractMatrix{T}, DimNames <: Tuple{Any, Any}, Inds <: Tuple{Any, Any}} = NamedDimsArray{
    T, 2, Denamed, DimNames, Inds
}

function NamedDimsArray(denamed::AbstractArray, dims)
    ndims(denamed) == length(dims) || throw(ArgumentError("Number of named dims must match ndims."))
    return NamedDimsArray(denamed, Tuple{Vararg{Any, ndims(denamed)}}(dims))
end
function NamedDimsArray(denamed::AbstractArray, ax::LittleDict)
    return NamedDimsArray(denamed, keys(ax), values(ax))
end

# TODO: Delete this, and just wrap the input naively.
function NamedDimsArray(a::AbstractNamedDimsArray, ax)
    return error("Already named.")
end

function NamedDimsArray(a::AbstractNamedDimsArray)
    return NamedDimsArray(denamed(a), axes(a))
end

# Minimal interface.
Base.axes(a::NamedDimsArray) = LittleDict(a.dimnames, a.inds)
denamed(a::NamedDimsArray) = a.denamed
Base.parent(a::NamedDimsArray) = a.denamed

function TypeParameterAccessors.position(
        ::Type{<:AbstractNamedDimsArray}, ::typeof(parenttype)
    )
    return TypeParameterAccessors.Position(3)
end
