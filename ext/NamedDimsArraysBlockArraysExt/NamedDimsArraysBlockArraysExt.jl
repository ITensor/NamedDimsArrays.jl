module NamedDimsArraysBlockArraysExt
using ArrayLayouts: ArrayLayouts
using BlockArrays: Block, BlockRange
using NamedDimsArrays:
  AbstractNamedDimsArray,
  AbstractNamedUnitRange,
  named_getindex,
  nameddims_getindex,
  nameddims_view

function Base.getindex(r::AbstractNamedUnitRange{<:Integer}, I::Block{1})
  # TODO: Use `Derive.@interface NamedArrayInterface() r[I]` instead.
  return named_getindex(r, I)
end

function Base.getindex(r::AbstractNamedUnitRange{<:Integer}, I::BlockRange{1})
  # TODO: Use `Derive.@interface NamedArrayInterface() r[I]` instead.
  return named_getindex(r, I)
end

const BlockIndex{N} = Union{Block{N},BlockRange{N},AbstractVector{<:Block{N}}}

function Base.view(a::AbstractNamedDimsArray, I1::Block{1}, Irest::BlockIndex{1}...)
  # TODO: Use `Derive.@interface NamedDimsArrayInterface() r[I]` instead.
  return nameddims_view(a, I1, Irest...)
end

function Base.view(a::AbstractNamedDimsArray, I::Block)
  # TODO: Use `Derive.@interface NamedDimsArrayInterface() r[I]` instead.
  return nameddims_view(a, Tuple(I)...)
end

function Base.view(a::AbstractNamedDimsArray, I1::BlockIndex{1}, Irest::BlockIndex{1}...)
  # TODO: Use `Derive.@interface NamedDimsArrayInterface() r[I]` instead.
  return nameddims_view(a, I1, Irest...)
end

# Fix ambiguity error.
function Base.getindex(
  a::AbstractNamedDimsArray, I1::BlockRange{1}, Irest::BlockRange{1}...
)
  return ArrayLayouts.layout_getindex(a, I1, Irest...)
end

end
