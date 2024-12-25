using TypeParameterAccessors: TypeParameterAccessors, parenttype

struct NamedDimsArray{T,N,Parent<:AbstractArray{T,N},DimNames} <:
       AbstractNamedDimsArray{T,N}
  parent::Parent
  dimnames::DimNames
end

const NamedDimsVector{T,Parent<:AbstractVector{T},DimNames} = NamedDimsArray{
  T,1,Parent,DimNames
}
const NamedDimsMatrix{T,Parent<:AbstractMatrix{T},DimNames} = NamedDimsArray{
  T,2,Parent,DimNames
}

function NamedDimsArray(a::AbstractNamedDimsArray, dimnames)
  return NamedDimsArray(denamed(a, dimnames), dimnames)
end

# Minimal interface.
dimnames(a::NamedDimsArray) = a.dimnames
Base.parent(a::NamedDimsArray) = a.parent

function TypeParameterAccessors.position(
  ::Type{<:AbstractNamedDimsArray}, ::typeof(parenttype)
)
  return TypeParameterAccessors.Position(3)
end
