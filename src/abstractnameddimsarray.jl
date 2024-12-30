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

Derive.interface(::Type{<:AbstractNamedDimsArray}) = NamedDimsArrayInterface()

# Output the dimension names.
dimnames(a::AbstractArray) = throw(MethodError(dimnames, Tuple{typeof(a)}))
# Unwrapping the names
Base.parent(a::AbstractNamedDimsArray) = throw(MethodError(parent, Tuple{typeof(a)}))

dimnames(a::AbstractArray, dim::Int) = dimnames(a)[dim]

dim(a::AbstractArray, n) = findfirst(==(name(n)), dimnames(a))
dims(a::AbstractArray, ns) = map(n -> dim(a, n), ns)

# TODO: Generalize to `AbstractNamedVector`.
dimname_isequal(x) = Base.Fix1(dimname_isequal, x)
dimname_isequal(x, y) = isequal(x, y)

dimname_isequal(r1::AbstractNamedUnitRange, r2::AbstractNamedUnitRange) = isequal(r1, r2)

dimname_isequal(r1::AbstractNamedUnitRange, r2) = name(r1) == r2
dimname_isequal(r1, r2::AbstractNamedUnitRange) = r1 == name(r2)

dimname_isequal(r1::AbstractNamedInteger, r2) = name(r1) == r2
dimname_isequal(r1, r2::AbstractNamedInteger) = r1 == name(r2)

function dimname_isequal(r1::AbstractNamedUnitRange, r2::AbstractNamedInteger)
  return isequal(r1, Base.oneto(r2))
end
function dimname_isequal(r1::AbstractNamedInteger, r2::AbstractNamedUnitRange)
  return isequal(Base.oneto(r1), r2)
end

dimname_isequal(r1::Name, r2) = dimname_isequal(name(r1), r2)
dimname_isequal(r1, r2::Name) = dimname_isequal(r1, name(r2))

function to_dimnames(a::AbstractArray, dims)
  return to_dimnames(a, axes(a), dims)
end
function to_dimnames(a::AbstractArray, axes, dims)
  return map((axis, dim) -> to_dimname(a, axis, dim), axes, dims)
end
# TODO: Generalize to `AbstractNamedVector`.
function to_dimname(a::AbstractArray, axis, dim::AbstractNamedUnitRange)
  # TODO: Check `axis == dim`?
  return name(dim)
end
function to_dimname(a::AbstractArray, axis, dim::AbstractNamedInteger)
  # TODO: Check `axis == Base.oneto(dim)`?
  return name(dim)
end
function to_dimname(a::AbstractArray, axis, dim::Name)
  return name(dim)
end
function to_dimname(a::AbstractArray, axis, dim)
  return dim
end

function to_dimnames(a::AbstractNamedDimsArray, axes, dims)
  perm = getperm(name.(axes), dims; isequal=dimname_isequal)
  all(!isnothing, perm) || throw(NameMismatch("Dimension name mismatch $(name.(axes)), $dims."))
  # Align the old names with the new names.
  # TODO: Check if `invperm` is correct here.
  axes_perm = map(p -> axes[p], invperm(perm))
  return map((axis, dim) -> to_dimname(a, axis, dim), axes_perm, dims)
end

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
  dimnames = to_dimnames(a, dims)
  # TODO: Check the shape of `dename.(dims)` matches the shape of `a`.
  # `mapreduce(typeof, promote_type, xs) == Base.promote_typeof(xs...)`.
  return nameddimstype(eltype(dimnames))(a, dimnames)
end
@traitfn function nameddims(a::AbstractArray::IsNamed, dims)
  return aligneddims(a, to_dimnames(a, dims))
end

function Base.view(a::AbstractArray, dimname1::AbstractName, dimname_rest::AbstractName...)
  return nameddims(a, (dimname1, dimname_rest...))
end
function Base.getindex(
  a::AbstractArray, dimname1::AbstractName, dimname_rest::AbstractName...
)
  return copy(@view(a[dimname1, dimname_rest...]))
end

# Fix ambiguity error.
function Base.view(a::AbstractNamedDimsArray, dimname1::AbstractName, dimname_rest::AbstractName...)
  return nameddims(a, (dimname1, dimname_rest...))
end
function Base.getindex(
  a::AbstractNamedDimsArray, dimname1::AbstractName, dimname_rest::AbstractName...
)
  return copy(@view(a[dimname1, dimname_rest...]))
end

Base.copy(a::AbstractNamedDimsArray) = nameddims(copy(dename(a)), dimnames(a))

# Can overload this to get custom named dims array wrapper
# depending on the dimension name types, for example
# output an `ITensor` if the dimension names are `IndexName`s.
nameddimstype(dimnametype::Type) = NamedDimsArray

Base.axes(a::AbstractNamedDimsArray) = map(named, axes(dename(a)), dimnames(a))
Base.size(a::AbstractNamedDimsArray) = map(named, size(dename(a)), dimnames(a))

# Circumvent issue when ndims isn't known at compile time.
function Base.axes(a::AbstractNamedDimsArray, d)
  return d <= ndims(a) ? axes(a)[d] : OneTo(1)
end

# Circumvent issue when ndims isn't known at compile time.
function Base.size(a::AbstractNamedDimsArray, d)
  return d <= ndims(a) ? size(a)[d] : OneTo(1)
end

# Circumvent issue when ndims isn't known at compile time.
Base.ndims(a::AbstractNamedDimsArray) = ndims(dename(a))

# Circumvent issue when eltype isn't known at compile time.
Base.eltype(a::AbstractNamedDimsArray) = eltype(dename(a))

Base.axes(a::AbstractNamedDimsArray, dimname::AbstractName) = axes(a, dim(a, dimname))
Base.size(a::AbstractNamedDimsArray, dimname::AbstractName) = size(a, dim(a, dimname))

function setdimnames(a::AbstractNamedDimsArray, dimnames)
  ## return nameddims(dename(a), to_dimnames(a, dimnames))
  return nameddims(dename(a), dimnames)
end
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
function getperm(x, y; isequal=isequal)
  return map(yᵢ -> findfirst(isequal(yᵢ), x), y)
end

# Indexing.

# Like `const ViewIndex = Union{Real,AbstractArray}`.
# TODO: Generalize to `AbstractNamedArray` for non-contiguous ranges.
const NamedViewIndex = Union{AbstractNamedInteger,AbstractNamedUnitRange}

function Base.getindex(a::AbstractNamedDimsArray, I1::Int, I_rest::Int...)
  return getindex(dename(a), I1, I_rest...)
end
function Base.getindex(a::AbstractNamedDimsArray, I::CartesianIndex)
  return getindex(a, to_indices(a, (I,))...)
end
function Base.getindex(
  a::AbstractNamedDimsArray, I1::NamedViewIndex, I_rest::NamedViewIndex...
)
  return getindex(a, to_indices(a, (I1, I_rest...))...)
end
function Base.getindex(a::AbstractNamedDimsArray, I::NamedCartesianIndex)
  return getindex(a, to_indices(a, (I,))...)
end
function Base.getindex(a::AbstractNamedDimsArray, I1::Pair, I_rest::Pair...)
  return getindex(a, to_indices(a, (I1, I_rest...))...)
end
function Base.getindex(a::AbstractNamedDimsArray)
  return getindex(dename(a))
end
# Linear indexing.
function Base.getindex(a::AbstractNamedDimsArray, I::Int)
  return getindex(dename(a), I)
end
function Base.setindex!(a::AbstractNamedDimsArray, value, I1::Int, I_rest::Int...)
  setindex!(dename(a), value, I1, I_rest...)
  return a
end
function Base.setindex!(
  a::AbstractNamedDimsArray, value, I1::NamedViewIndex, I_rest::NamedViewIndex...
)
  setindex!(a, value, to_indices(a, (I1, I_rest...))...)
  return a
end
function Base.setindex!(a::AbstractNamedDimsArray, value, I::NamedCartesianIndex)
  setindex!(a, value, to_indices(a, (I,))...)
  return a
end
function Base.setindex!(a::AbstractNamedDimsArray, value, I1::Pair, I_rest::Pair...)
  setindex!(a, value, to_indices(a, (I1, I_rest...))...)
  return a
end
function Base.setindex!(a::AbstractNamedDimsArray, value)
  setindex!(dename(a), value)
  return a
end
# Linear indexing.
function Base.setindex!(a::AbstractNamedDimsArray, value, I::Int)
  setindex!(dename(a), value, I)
  return a
end
# Handles permutation of indices to align dimension names.
function Base.to_indices(
  a::AbstractNamedDimsArray, I::Tuple{NamedViewIndex,Vararg{NamedViewIndex}}
)
  # TODO: Check this permutation is correct (it may be the inverse of what we want).
  return dename.(map(i -> I[i], getperm(dimnames(a), name.(I); isequal=dimname_isequal)))
end
function Base.to_indices(a::AbstractNamedDimsArray, I::Tuple{NamedCartesianIndex})
  return to_indices(a, Tuple(only(I)))
end
# Support indexing with pairs `a[:i => 1, :j => 2]`.
function Base.to_indices(a::AbstractNamedDimsArray, I::Tuple{Pair,Vararg{Pair}})
  return to_indices(a, named.(last.(I), first.(I)))
end
function Base.isassigned(a::AbstractNamedDimsArray, I::Vararg{Int})
  return isassigned(parent(a), I...)
end

# Slicing

using ArrayLayouts: ArrayLayouts
using Derive: @derive

@derive (T=AbstractNamedDimsArray,) begin
  Base.getindex(::T, ::Any...)
  Base.setindex!(::T, ::Any, ::Any...)
end

function Base.view(a::AbstractNamedDimsArray, I...)
  I′ = to_indices(a, I)
  return nameddims(view(parent(a), I′...), map((n, i) -> n[i], dimnames(a), I′))
end

function aligndims(a::AbstractArray, dims)
  new_dimnames = to_dimnames(a, dims)
  # TODO: Check this permutation is correct (it may be the inverse of what we want).
  perm = getperm(dimnames(a), new_dimnames)
  return nameddims(permutedims(dename(a), perm), new_dimnames)
end

function aligneddims(a::AbstractArray, dims)
  # TODO: Check this permutation is correct (it may be the inverse of what we want).
  new_dimnames = to_dimnames(a, dims)
  perm = getperm(dimnames(a), new_dimnames)
  !isperm(perm) &&
    throw(NameMismatch("Dimension name mismatch $(dimnames(a)), $(new_dimnames)."))
  return nameddims(PermutedDimsArray(dename(a), perm), new_dimnames)
end

using Random: Random, AbstractRNG

# TODO: Come up with a better name for this.
_rand(args...) = Base.rand(args...)
function _rand(
  rng::AbstractRNG, elt::Type, dims::Tuple{Base.OneTo{Int},Vararg{Base.OneTo{Int}}}
)
  return Base.rand(rng, elt, length.(dims))
end

# TODO: Come up with a better name for this.
_randn(args...) = Base.randn(args...)
function _randn(
  rng::AbstractRNG, elt::Type, dims::Tuple{Base.OneTo{Int},Vararg{Base.OneTo{Int}}}
)
  return Base.randn(rng, elt, length.(dims))
end

# Convenient constructors
default_eltype() = Float64
for (f, f′) in [(:rand, :_rand), (:randn, :_randn)]
  @eval begin
    function Base.$f(
      rng::AbstractRNG,
      elt::Type{<:Number},
      dims::Tuple{AbstractNamedUnitRange,Vararg{AbstractNamedUnitRange}},
    )
      a = $f′(rng, elt, unname.(dims))
      return nameddims(a, dims)
    end
    function Base.$f(
      rng::AbstractRNG,
      elt::Type{<:Number},
      dims::Tuple{AbstractNamedInteger,Vararg{AbstractNamedInteger}},
    )
      return $f(rng, elt, Base.oneto.(dims))
    end
  end
  for dimtype in [:AbstractNamedInteger, :AbstractNamedUnitRange]
    @eval begin
      function Base.$f(
        rng::AbstractRNG, elt::Type{<:Number}, dim1::$dimtype, dims::Vararg{$dimtype}
      )
        return $f(rng, elt, (dim1, dims...))
      end
      Base.$f(elt::Type{<:Number}, dims::Tuple{$dimtype,Vararg{$dimtype}}) =
        $f(Random.default_rng(), elt, dims)
      Base.$f(elt::Type{<:Number}, dim1::$dimtype, dims::Vararg{$dimtype}) =
        $f(elt, (dim1, dims...))
      Base.$f(dims::Tuple{$dimtype,Vararg{$dimtype}}) = $f(default_eltype(), dims)
      Base.$f(dim1::$dimtype, dims::Vararg{$dimtype}) = $f((dim1, dims...))
    end
  end
end
for f in [:zeros, :ones]
  for dimtype in [:AbstractNamedInteger, :AbstractNamedUnitRange]
    @eval begin
      function Base.$f(elt::Type{<:Number}, dims::Tuple{$dimtype,Vararg{$dimtype}})
        a = $f(elt, unname.(dims))
        return nameddims(a, to_dimnames(a, dims))
      end
      function Base.$f(elt::Type{<:Number}, dim1::$dimtype, dims::Vararg{$dimtype})
        return $f(elt, (dim1, dims...))
      end
      Base.$f(dims::Tuple{$dimtype,Vararg{$dimtype}}) = $f(default_eltype(), dims)
      Base.$f(dim1::$dimtype, dims::Vararg{$dimtype}) = $f((dim1, dims...))
    end
  end
end
for dimtype in [:AbstractNamedInteger, :AbstractNamedUnitRange]
  @eval begin
    function Base.fill(value, dims::Tuple{$dimtype,Vararg{$dimtype}})
      a = fill(value, unname.(dims))
      return nameddims(a, to_dimnames(a, dims))
    end
    function Base.fill(value, dim1::$dimtype, dims::Vararg{$dimtype})
      return fill(value, (dim1, dims...))
    end
  end
end

using Base.Broadcast:
  AbstractArrayStyle,
  Broadcasted,
  broadcast_shape,
  broadcasted,
  check_broadcast_shape,
  combine_axes
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
  perm = getperm(ax1, ax2)
  ax2_aligned = map(i -> ax2[i], perm)
  ax_promoted = promote_shape(dename.(ax1), dename.(ax2_aligned))
  return named.(ax_promoted, name.(ax1))
end

# Avoid comparison of `NamedInteger` against `1`.
function Broadcast.check_broadcast_shape(
  ax1::Tuple{AbstractNamedUnitRange,Vararg{AbstractNamedUnitRange,N}},
  ax2::Tuple{AbstractNamedUnitRange,Vararg{AbstractNamedUnitRange,N}},
) where {N}
  perm = getperm(ax1, ax2)
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
  m′ = denamed(Mapped(bc), ax)
  return nameddims(similar(m′, elt, dename.(ax)), ax)
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
