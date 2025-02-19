using TypeParameterAccessors: TypeParameterAccessors, parenttype

# nameddimsindices should be a named slice.
struct NamedDimsArray{T,N,Parent<:AbstractArray{T,N},DimNames} <:
       AbstractNamedDimsArray{T,N}
  parent::Parent
  nameddimsindices::DimNames
  function NamedDimsArray(parent::AbstractArray, dims)
    # This checks the shapes of the inputs.
    nameddimsindices = to_nameddimsindices(parent, dims)
    return new{eltype(parent),ndims(parent),typeof(parent),typeof(nameddimsindices)}(
      parent, nameddimsindices
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
function NamedDimsArray(a::AbstractNamedDimsArray, nameddimsindices)
  return error("Already named.")
end

function NamedDimsArray(a::AbstractNamedDimsArray)
  return NamedDimsArray(dename(a), nameddimsindices(a))
end

# Minimal interface.
nameddimsindices(a::NamedDimsArray) = a.nameddimsindices
Base.parent(a::NamedDimsArray) = a.parent

function TypeParameterAccessors.position(
  ::Type{<:AbstractNamedDimsArray}, ::typeof(parenttype)
)
  return TypeParameterAccessors.Position(3)
end
