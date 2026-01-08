using TypeParameterAccessors: TypeParameterAccessors, parenttype

# inds should be a named slice.
struct NamedDimsArray{T, N, Parent <: AbstractArray{T, N}, DimNames} <:
    AbstractNamedDimsArray{T, N}
    parent::Parent
    inds::DimNames
    function NamedDimsArray(parent::AbstractArray, dims)
        # This checks the shapes of the inputs.
        inds = to_inds(parent, dims)
        return new{eltype(parent), ndims(parent), typeof(parent), typeof(inds)}(
            parent, inds
        )
    end
end

const NamedDimsVector{T, Parent <: AbstractVector{T}, DimNames} = NamedDimsArray{
    T, 1, Parent, DimNames,
}
const NamedDimsMatrix{T, Parent <: AbstractMatrix{T}, DimNames} = NamedDimsArray{
    T, 2, Parent, DimNames,
}

# TODO: Delete this, and just wrap the input naively.
function NamedDimsArray(a::AbstractNamedDimsArray, inds)
    return error("Already named.")
end

function NamedDimsArray(a::AbstractNamedDimsArray)
    return NamedDimsArray(denamed(a), inds(a))
end

# Minimal interface.
inds(a::NamedDimsArray) = getfield(a, :inds)
Base.parent(a::NamedDimsArray) = getfield(a, :parent)
denamed(a::NamedDimsArray) = parent(a)

function TypeParameterAccessors.position(
        ::Type{<:AbstractNamedDimsArray}, ::typeof(parenttype)
    )
    return TypeParameterAccessors.Position(3)
end
