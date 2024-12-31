using TypeParameterAccessors: TypeParameterAccessors, parenttype

# dimnames should be a named slice.
struct NamedDimsArray{T,N,Parent<:AbstractArray{T,N},DimNames} <:
       AbstractNamedDimsArray{T,N}
  parent::Parent
  dimnames::DimNames
  function NamedDimsArray(parent::AbstractArray, dims)
    dimnames = to_dimnames(parent, dims)
    return new{eltype(parent),ndims(parent),typeof(parent),typeof(dimnames)}(
      parent, dimnames
    )
  end
end

const NamedDimsVector{T,Parent<:AbstractVector{T},DimNames} = NamedDimsArray{
  T,1,Parent,DimNames
}
const NamedDimsMatrix{T,Parent<:AbstractMatrix{T},DimNames} = NamedDimsArray{
  T,2,Parent,DimNames
}

# TODO: Delete this, and just wrap the input naively.
function NamedDimsArray(a::AbstractNamedDimsArray, dimnames)
  return error("Already named.")
end

function NamedDimsArray(a::AbstractNamedDimsArray)
  return NamedDimsArray(dename(a), dimnames(a))
end

# Minimal interface.
dimnames(a::NamedDimsArray) = a.dimnames
Base.parent(a::NamedDimsArray) = a.parent

function TypeParameterAccessors.position(
  ::Type{<:AbstractNamedDimsArray}, ::typeof(parenttype)
)
  return TypeParameterAccessors.Position(3)
end
