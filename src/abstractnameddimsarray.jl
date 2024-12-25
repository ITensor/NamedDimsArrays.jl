using Derive: Derive, @derive, AbstractArrayInterface

# Some of the interface is inspired by:
# https://github.com/ITensor/ITensors.jl
# https://github.com/invenia/NamedDims.jl
# https://github.com/mcabbott/NamedPlus.jl

abstract type AbstractNamedDimsArrayInterface <: AbstractArrayInterface end

struct NamedDimsArrayInterface <: AbstractNamedDimsArrayInterface end

abstract type AbstractNamedDimsArray{T,N} <: AbstractArray{T,N} end

const AbstractNamedDimsVector{T} = AbstractNamedDimsArray{T,1}
const AbstractNamedDimsMatrix{T} = AbstractNamedDimsArray{T,2}

Derive.interface(::Type{<:AbstractNamedDimsArray}) = AbstractNamedDimsArrayInterface()

# Output the dimension names.
dimnames(a::AbstractArray) = throw(MethodError(dimnames, Tuple{typeof(a)}))
# Unwrapping the names
Base.parent(a::AbstractNamedDimsArray) = throw(MethodError(parent, Tuple{typeof(a)}))

dimnames(a::AbstractArray, dim::Int) = dimnames(a)[dim]

dim(a::AbstractArray, n) = findfirst(==(name(n)), dimnames(a))
dims(a::AbstractArray, ns) = map(n -> dim(a, n), ns)

# Unwrapping the names (`NamedDimsArrays.jl` interface).
# TODO: Use `IsNamed` trait?
dename(a::AbstractNamedDimsArray) = parent(a)
function dename(a::AbstractNamedDimsArray, dimnames)
  return dename(aligndims(a, dimnames))
end
function denamed(a::AbstractNamedDimsArray, dimnames)
  return dename(aligneddims(a, dimnames))
end

unname(a::AbstractArray, dimnames) = dename(a, dimnames)
unnamed(a::AbstractArray, dimnames) = denamed(a, dimnames)

isnamed(::Type{<:AbstractNamedDimsArray}) = true

# Can overload this to get custom named dims array wrapper
# depending on the dimension name types, for example
# output an `ITensor` if the dimension names are `IndexName`s.
@traitfn function nameddims(a::AbstractArray::!(IsNamed), dims)
  dimnames = name.(dims)
  # TODO: Check the shape of `dename.(dims)` matches the shape of `a`.
  # `mapreduce(typeof, promote_type, xs) == Base.promote_typeof(xs...)`.
  return nameddimstype(eltype(dimnames))(a, dimnames)
end
@traitfn function nameddims(a::AbstractArray::IsNamed, dims)
  return aligneddims(a, dims)
end

function Base.view(a::AbstractArray, dimnames::AbstractName...)
  return nameddims(a, dimnames)
end
function Base.getindex(a::AbstractArray, dimnames::AbstractName...)
  return copy(@view(a[dimnames...]))
end

Base.copy(a::AbstractNamedDimsArray) = nameddims(copy(dename(a)), dimnames(a))

# Can overload this to get custom named dims array wrapper
# depending on the dimension name types, for example
# output an `ITensor` if the dimension names are `IndexName`s.
nameddimstype(dimnametype::Type) = NamedDimsArray

Base.axes(a::AbstractNamedDimsArray) = map(named, axes(dename(a)), dimnames(a))
Base.size(a::AbstractNamedDimsArray) = map(named, size(dename(a)), dimnames(a))

Base.axes(a::AbstractArray, dimname::AbstractName) = axes(a, dim(a, dimname))
Base.size(a::AbstractArray, dimname::AbstractName) = size(a, dim(a, dimname))

setdimnames(a::AbstractNamedDimsArray, dimnames) = nameddims(dename(a), name.(dimnames))
function replacedimnames(f, a::AbstractNamedDimsArray)
  return setdimnames(a, replace(f, dimnames(a)))
end
function replacedimnames(a::AbstractNamedDimsArray, replacements::Pair...)
  replacement_names = map(replacements) do replacement
    name(first(replacement)) => name(last(replacement))
  end
  new_dimnames = replace(dimnames(a), replacement_names...)
  return setdimnames(a, new_dimnames)
end

# `Base.isempty(a::AbstractArray)` is defined as `length(a) == 0`,
# which involves comparing a named integer to an unnamed integer
# which isn't well defined.
Base.isempty(a::AbstractNamedDimsArray) = isempty(dename(a))

# Define this on objects rather than types in case the wrapper type
# isn't known at compile time, like for the ITensor type.
Base.IndexStyle(a::AbstractNamedDimsArray) = IndexStyle(dename(a))
Base.eachindex(a::AbstractNamedDimsArray) = eachindex(dename(a))

# Cartesian indices with names attached.
struct NamedIndexCartesian <: IndexStyle end

# When multiple named dims arrays are involved, use the named
# dimensions.
function Base.IndexStyle(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
  return NamedIndexCartesian()
end
# Define promotion of index styles.
Base.IndexStyle(s1::NamedIndexCartesian, s2::NamedIndexCartesian) = NamedIndexCartesian()
Base.IndexStyle(s1::IndexStyle, s2::NamedIndexCartesian) = NamedIndexCartesian()
Base.IndexStyle(s1::NamedIndexCartesian, s2::IndexStyle) = NamedIndexCartesian()

# Like CartesianIndex but with named dimensions.
struct NamedCartesianIndex{N,Index<:Tuple{Vararg{AbstractNamedInteger,N}}} <:
       Base.AbstractCartesianIndex{N}
  I::Index
end
NamedCartesianIndex(I::AbstractNamedInteger...) = NamedCartesianIndex(I)
Base.Tuple(I::NamedCartesianIndex) = I.I
function Base.show(io::IO, I::NamedCartesianIndex)
  print(io, "NamedCartesianIndex")
  show(io, Tuple(I))
  return nothing
end

# Like CartesianIndices but with named dimensions.
struct NamedCartesianIndices{
  N,
  Indices<:Tuple{Vararg{AbstractNamedUnitRange,N}},
  Index<:Tuple{Vararg{AbstractNamedInteger,N}},
} <: AbstractNamedDimsArray{NamedCartesianIndex{N,Index},N}
  indices::Indices
  function NamedCartesianIndices(indices::Tuple{Vararg{AbstractNamedUnitRange}})
    return new{length(indices),typeof(indices),Tuple{eltype.(indices)...}}(indices)
  end
end

Base.axes(I::NamedCartesianIndices) = map(only ∘ axes, I.indices)
Base.size(I::NamedCartesianIndices) = length.(I.indices)

function Base.getindex(a::NamedCartesianIndices{N}, I::Vararg{Int,N}) where {N}
  index = map(a.indices, I) do r, i
    return getindex(r, i)
  end
  return NamedCartesianIndex(index)
end

dimnames(I::NamedCartesianIndices) = name.(I.indices)
function Base.parent(I::NamedCartesianIndices)
  return CartesianIndices(dename.(I.indices))
end

function Base.eachindex(::NamedIndexCartesian, a1::AbstractArray, a_rest::AbstractArray...)
  all(a -> issetequal(dimnames(a1), dimnames(a)), a_rest) ||
    throw(NameMismatch("Dimension name mismatch $(dimnames.((a1, a_rest...)))."))
  # TODO: Check the shapes match.
  return NamedCartesianIndices(axes(a1))
end

# Base version ignores dimension names.
# TODO: Use `mapreduce(isequal, &&, a1, a2)`?
function Base.isequal(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
  return all(eachindex(a1, a2)) do I
    isequal(a1[I], a2[I])
  end
end

# Base version ignores dimension names.
# TODO: Use `mapreduce(==, &&, a1, a2)`?
# TODO: Handle `missing` values properly.
function Base.:(==)(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
  return all(eachindex(a1, a2)) do I
    a1[I] == a2[I]
  end
end

# TODO: Move to `utils.jl` file.
# TODO: Use `Base.indexin`?
function getperm(x, y)
  return map(yᵢ -> findfirst(isequal(yᵢ), x), y)
end

# Indexing.
function Base.getindex(a::AbstractNamedDimsArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return getindex(dename(a), I...)
end
function Base.getindex(
  a::AbstractArray{<:Any,N}, I::Vararg{AbstractNamedInteger,N}
) where {N}
  return getindex(a, to_indices(a, I)...)
end
function Base.getindex(
  a::AbstractNamedDimsArray{<:Any,N}, I::NamedCartesianIndex{N}
) where {N}
  return getindex(a, to_indices(a, (I,))...)
end
function Base.getindex(
  a::AbstractNamedDimsArray{<:Any,N}, I::Vararg{Pair{<:Any,Int},N}
) where {N}
  return getindex(a, to_indices(a, I)...)
end
function Base.getindex(a::AbstractNamedDimsArray, I::Int)
  return getindex(dename(a), I)
end
function Base.setindex!(
  a::AbstractNamedDimsArray{<:Any,N}, value, I::Vararg{Int,N}
) where {N}
  setindex!(dename(a), value, I...)
  return a
end
function Base.setindex!(
  a::AbstractArray{<:Any,N}, value, I::Vararg{AbstractNamedInteger,N}
) where {N}
  setindex!(a, value, to_indices(a, I)...)
  return a
end
function Base.setindex!(
  a::AbstractNamedDimsArray{<:Any,N}, value, I::NamedCartesianIndex{N}
) where {N}
  setindex!(a, value, to_indices(a, (I,))...)
  return a
end
function Base.setindex!(
  a::AbstractNamedDimsArray{<:Any,N}, value, I::Vararg{Pair{<:Any,Int},N}
) where {N}
  setindex!(a, value, to_indices(a, I)...)
  return a
end
function Base.setindex!(a::AbstractNamedDimsArray, value, I::Int)
  setindex!(dename(a), value, I)
  return a
end
# Handles permutation of indices to align dimension names.
function Base.to_indices(
  a::AbstractArray{<:Any,N}, I::Tuple{Vararg{AbstractNamedInteger,N}}
) where {N}
  # TODO: Check this permutation is correct (it may be the inverse of what we want).
  # We unwrap the names twice in case named axes were passed as indices.
  return dename.(map(i -> I[i], getperm(dimnames(a), name.(name.(I)))))
end
function Base.to_indices(
  a::AbstractArray{<:Any,N}, I::Tuple{NamedCartesianIndex{N}}
) where {N}
  return to_indices(a, Tuple(only(I)))
end
# Support indexing with pairs `a[:i => 1, :j => 2]`.
function Base.to_index(a::AbstractNamedDimsArray, i::Pair{<:Any,Int})
  return named(last(i), first(i))
end
function Base.isassigned(a::AbstractNamedDimsArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return isassigned(parent(a), I...)
end

function aligndims(a::AbstractArray, dims)
  # TODO: Check this permutation is correct (it may be the inverse of what we want).
  perm = getperm(dimnames(a), name.(dims))
  return nameddims(permutedims(dename(a), perm), name.(dims))
end

function aligneddims(a::AbstractArray, dims)
  # TODO: Check this permutation is correct (it may be the inverse of what we want).
  new_dimnames = name.(dims)
  perm = getperm(dimnames(a), new_dimnames)
  !isperm(perm) &&
    throw(NameMismatch("Dimension name mismatch $(dimnames(a)), $(new_dimnames)."))
  return nameddims(PermutedDimsArray(dename(a), perm), new_dimnames)
end

using Random: Random, AbstractRNG

# Convenient constructors
default_eltype() = Float64
for f in [:rand, :randn]
  @eval begin
    function Base.$f(
      rng::AbstractRNG,
      elt::Type{<:Number},
      dims::Tuple{AbstractNamedInteger,Vararg{AbstractNamedInteger}},
    )
      a = $f(rng, elt, unname.(dims))
      return nameddims(a, name.(dims))
    end
    function Base.$f(
      rng::AbstractRNG,
      elt::Type{<:Number},
      dim1::AbstractNamedInteger,
      dims::Vararg{AbstractNamedInteger},
    )
      return $f(rng, elt, (dim1, dims...))
    end
    Base.$f(
      elt::Type{<:Number}, dims::Tuple{AbstractNamedInteger,Vararg{AbstractNamedInteger}}
    ) = $f(Random.default_rng(), elt, dims)
    Base.$f(
      elt::Type{<:Number}, dim1::AbstractNamedInteger, dims::Vararg{AbstractNamedInteger}
    ) = $f(elt, (dim1, dims...))
    Base.$f(dims::Tuple{AbstractNamedInteger,Vararg{AbstractNamedInteger}}) =
      $f(default_eltype(), dims)
    Base.$f(dim1::AbstractNamedInteger, dims::Vararg{AbstractNamedInteger}) =
      $f((dim1, dims...))
  end
end
for f in [:zeros, :ones]
  @eval begin
    function Base.$f(
      elt::Type{<:Number}, dims::Tuple{AbstractNamedInteger,Vararg{AbstractNamedInteger}}
    )
      a = $f(elt, unname.(dims))
      return nameddims(a, name.(dims))
    end
    function Base.$f(
      elt::Type{<:Number}, dim1::AbstractNamedInteger, dims::Vararg{AbstractNamedInteger}
    )
      return $f(elt, (dim1, dims...))
    end
    Base.$f(dims::Tuple{AbstractNamedInteger,Vararg{AbstractNamedInteger}}) =
      $f(default_eltype(), dims)
    Base.$f(dim1::AbstractNamedInteger, dims::Vararg{AbstractNamedInteger}) =
      $f((dim1, dims...))
  end
end
function Base.fill(value, dims::Tuple{AbstractNamedInteger,Vararg{AbstractNamedInteger}})
  a = fill(value, unname.(dims))
  return nameddims(a, name.(dims))
end
function Base.fill(value, dim1::AbstractNamedInteger, dims::Vararg{AbstractNamedInteger})
  return fill(value, (dim1, dims...))
end

using Base.Broadcast:
  AbstractArrayStyle,
  Broadcasted,
  broadcast_shape,
  broadcasted,
  check_broadcast_shape,
  combine_axes,
  combine_eltypes
using BroadcastMapConversion: Mapped, mapped

abstract type AbstractNamedDimsArrayStyle{N} <: AbstractArrayStyle{N} end

struct NamedDimsArrayStyle{N} <: AbstractNamedDimsArrayStyle{N} end
NamedDimsArrayStyle(::Val{N}) where {N} = NamedDimsArrayStyle{N}()
NamedDimsArrayStyle{M}(::Val{N}) where {M,N} = NamedDimsArrayStyle{N}()

function Broadcast.BroadcastStyle(arraytype::Type{<:AbstractNamedDimsArray})
  return NamedDimsArrayStyle{ndims(arraytype)}()
end

function Broadcast.combine_axes(
  a1::AbstractNamedDimsArray, a_rest::AbstractNamedDimsArray...
)
  return broadcast_shape(axes(a1), combine_axes(a_rest...))
end
function Broadcast.combine_axes(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
  return broadcast_shape(axes(a1), axes(a2))
end
Broadcast.combine_axes(a::AbstractNamedDimsArray) = axes(a)

function Broadcast.broadcast_shape(
  ax1::Tuple{Vararg{AbstractNamedUnitRange}},
  ax2::Tuple{Vararg{AbstractNamedUnitRange}},
  ax_rest::Tuple{Vararg{AbstractNamedUnitRange}}...,
)
  return broadcast_shape(broadcast_shape(ax1, ax2), ax_rest...)
end

function Broadcast.broadcast_shape(
  ax1::Tuple{Vararg{AbstractNamedUnitRange}}, ax2::Tuple{Vararg{AbstractNamedUnitRange}}
)
  return promote_shape(ax1, ax2)
end

function Base.promote_shape(
  ax1::Tuple{AbstractNamedUnitRange,Vararg{AbstractNamedUnitRange,N}},
  ax2::Tuple{AbstractNamedUnitRange,Vararg{AbstractNamedUnitRange,N}},
) where {N}
  perm = getperm(name.(ax1), name.(ax2))
  ax2_aligned = map(i -> ax2[i], perm)
  ax_promoted = promote_shape(dename.(ax1), dename.(ax2_aligned))
  return named.(ax_promoted, name.(ax1))
end

# Avoid comparison of `NamedInteger` against `1`.
function Broadcast.check_broadcast_shape(
  ax1::Tuple{AbstractNamedUnitRange,Vararg{AbstractNamedUnitRange,N}},
  ax2::Tuple{AbstractNamedUnitRange,Vararg{AbstractNamedUnitRange,N}},
) where {N}
  perm = getperm(name.(ax1), name.(ax2))
  ax2_aligned = map(i -> ax2[i], perm)
  check_broadcast_shape(dename.(ax1), dename.(ax2_aligned))
  return nothing
end

# Handle scalars.
function Base.promote_shape(
  ax1::Tuple{AbstractNamedUnitRange,Vararg{AbstractNamedUnitRange}}, ax2::Tuple{}
)
  return ax1
end

# Dename and lazily permute the arguments using the reference
# dimension names.
# TODO: Make a version that gets the dimnames from `m`.
function denamed(m::Mapped, dimnames)
  return mapped(m.f, map(arg -> denamed(arg, dimnames), m.args)...)
end

function Base.similar(bc::Broadcasted{<:AbstractNamedDimsArrayStyle}, elt::Type, ax::Tuple)
  m′ = denamed(Mapped(bc), name.(ax))
  return nameddims(similar(m′, elt, dename.(ax)), name.(ax))
end

function Base.copyto!(
  dest::AbstractArray{<:Any,N}, bc::Broadcasted{<:AbstractNamedDimsArrayStyle{N}}
) where {N}
  return copyto!(dest, Mapped(bc))
end

function Base.map!(f, a_dest::AbstractNamedDimsArray, a_srcs::AbstractNamedDimsArray...)
  a′_dest = dename(a_dest)
  # TODO: Use `denamed` to do the permutations lazily.
  a′_srcs = map(a_src -> dename(a_src, dimnames(a_dest)), a_srcs)
  map!(f, a′_dest, a′_srcs...)
  return a_dest
end

function Base.map(f, a_srcs::AbstractNamedDimsArray...)
  # copy(mapped(f, a_srcs...))
  return f.(a_srcs...)
end

function Base.mapreduce(f, op, a::AbstractNamedDimsArray; kwargs...)
  return mapreduce(f, op, dename(a); kwargs...)
end

using LinearAlgebra: LinearAlgebra, norm
function LinearAlgebra.norm(a::AbstractNamedDimsArray; kwargs...)
  return norm(dename(a); kwargs...)
end

# Printing.
function Base.show(io::IO, mime::MIME"text/plain", a::AbstractNamedDimsArray)
  summary(io, a)
  println(io)
  show(io, mime, dename(a))
  return nothing
end

function Base.show(io::IO, a::AbstractNamedDimsArray)
  print(io, "nameddims(")
  show(io, dename(a))
  print(io, ", ", dimnames(a), ")")
  return nothing
end
