using TypeParameterAccessors: TypeParameterAccessors, parenttype

# axes should be a named slice.
struct NamedDimsArray{T, N, Parent <: AbstractArray{T, N}, DimNames <: Tuple{Vararg{Any, N}}} <:
    AbstractNamedDimsArray{T, N}
    parent::Parent
    axes::DimNames
    function NamedDimsArray{T, N, Parent, DimNames}(
            parent::AbstractArray{<:Any, N}, dims::Tuple{Vararg{Any, N}}
        ) where {T, N, Parent <: AbstractArray{T, N}, DimNames <: Tuple{Vararg{Any, N}}}
        ax = to_axes(parent, dims)
        return new{T, N, Parent, DimNames}(parent, ax)
    end
end
function NamedDimsArray(
        parent::Parent, dims::Tuple{Vararg{Any, N}}
    ) where {T, N, Parent <: AbstractArray{T, N}}
    # This checks the shapes of the inputs.
    ax = to_axes(parent, dims)
    return NamedDimsArray{T, N, Parent, typeof(ax)}(parent, ax)
end

const NamedDimsVector{T, Parent <: AbstractVector{T}, DimNames <: Tuple{Any}} = NamedDimsArray{
    T, 1, Parent, DimNames,
}
const NamedDimsMatrix{T, Parent <: AbstractMatrix{T}, DimNames <: Tuple{Any, Any}} = NamedDimsArray{
    T, 2, Parent, DimNames,
}

function NamedDimsArray(parent::AbstractArray, dims)
    ndims(parent) == length(dims) || throw(ArgumentError("Number of named dims must match ndims."))
    return NamedDimsArray(parent, Tuple{Vararg{Any, ndims(parent)}}(dims))
end

# TODO: Delete this, and just wrap the input naively.
function NamedDimsArray(a::AbstractNamedDimsArray, ax)
    return error("Already named.")
end

function NamedDimsArray(a::AbstractNamedDimsArray)
    return NamedDimsArray(denamed(a), axes(a))
end

# Minimal interface.
Base.axes(a::NamedDimsArray) = getfield(a, :axes)
Base.parent(a::NamedDimsArray) = getfield(a, :parent)
denamed(a::NamedDimsArray) = parent(a)

function TypeParameterAccessors.position(
        ::Type{<:AbstractNamedDimsArray}, ::typeof(parenttype)
    )
    return TypeParameterAccessors.Position(3)
end
