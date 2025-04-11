module NamedDimsArraysBlockArraysExt
using ArrayLayouts: ArrayLayouts
using BlockArrays: Block, BlockRange
using NamedDimsArrays:
  AbstractNamedDimsArray,
  AbstractNamedDimsMatrix,
  AbstractNamedUnitRange,
  getindex_named,
  view_nameddimsarray

function Base.getindex(r::AbstractNamedUnitRange{<:Integer}, I::Block{1})
  # TODO: Use `Derive.@interface NamedArrayInterface() r[I]` instead.
  return getindex_named(r, I)
end

function Base.getindex(r::AbstractNamedUnitRange{<:Integer}, I::BlockRange{1})
  # TODO: Use `Derive.@interface NamedArrayInterface() r[I]` instead.
  return getindex_named(r, I)
end

const BlockIndex{N} = Union{Block{N},BlockRange{N},AbstractVector{<:Block{N}}}

function Base.view(a::AbstractNamedDimsArray, I1::Block{1}, Irest::BlockIndex{1}...)
  # TODO: Use `Derive.@interface NamedDimsArrayInterface() r[I]` instead.
  return view_nameddimsarray(a, I1, Irest...)
end

function Base.view(a::AbstractNamedDimsArray, I::Block)
  # TODO: Use `Derive.@interface NamedDimsArrayInterface() r[I]` instead.
  return view_nameddimsarray(a, Tuple(I)...)
end

function Base.view(a::AbstractNamedDimsArray, I1::BlockIndex{1}, Irest::BlockIndex{1}...)
  # TODO: Use `Derive.@interface NamedDimsArrayInterface() r[I]` instead.
  return view_nameddimsarray(a, I1, Irest...)
end

# Fix ambiguity error.
function Base.getindex(
  a::AbstractNamedDimsArray, I1::BlockRange{1}, Irest::BlockRange{1}...
)
  return ArrayLayouts.layout_getindex(a, I1, Irest...)
end

# Fix ambiguity errors.
function Base.getindex(a::AbstractNamedDimsArray, I1::Block{1}, Irest...)
  return copy(view(a, I1, Irest...))
end
function Base.getindex(a::AbstractNamedDimsMatrix, I1::AbstractVector, I2::Block{1})
  return copy(view(a, I1, I2))
end
function Base.getindex(a::AbstractNamedDimsMatrix, I1::Block{1}, I2::AbstractVector)
  return copy(view(a, I1, I2))
end
function Base.getindex(a::AbstractNamedDimsArray{<:Any,N}, I::Block{N}) where {N}
  return copy(view(a, I))
end

end
