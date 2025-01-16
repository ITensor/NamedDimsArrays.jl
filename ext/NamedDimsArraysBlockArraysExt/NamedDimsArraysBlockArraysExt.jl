module NamedDimsArraysBlockArraysExt
using ArrayLayouts: ArrayLayouts
using BlockArrays: Block, BlockRange
using NamedDimsArrays:
  AbstractNamedDimsArray, AbstractNamedUnitRange, getindex_named, view_nameddims

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
  return view_nameddims(a, I1, Irest...)
end

function Base.view(a::AbstractNamedDimsArray, I::Block)
  # TODO: Use `Derive.@interface NamedDimsArrayInterface() r[I]` instead.
  return view_nameddims(a, Tuple(I)...)
end

function Base.view(a::AbstractNamedDimsArray, I1::BlockIndex{1}, Irest::BlockIndex{1}...)
  # TODO: Use `Derive.@interface NamedDimsArrayInterface() r[I]` instead.
  return view_nameddims(a, I1, Irest...)
end

# Fix ambiguity error.
function Base.getindex(
  a::AbstractNamedDimsArray, I1::BlockRange{1}, Irest::BlockRange{1}...
)
  return ArrayLayouts.layout_getindex(a, I1, Irest...)
end

end
