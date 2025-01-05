using Derive: Derive, @derive, AbstractArrayInterface

# Some of the interface is inspired by:
# https://github.com/ITensor/ITensors.jl
# https://github.com/invenia/NamedDims.jl
# https://github.com/mcabbott/NamedPlus.jl
# https://pytorch.org/docs/stable/named_tensor.html

abstract type AbstractNamedDimsArrayInterface <: AbstractArrayInterface end

struct NamedDimsArrayInterface <: AbstractNamedDimsArrayInterface end

abstract type AbstractNamedDimsArray{T,N} <: AbstractArray{T,N} end

const AbstractNamedDimsVector{T} = AbstractNamedDimsArray{T,1}
const AbstractNamedDimsMatrix{T} = AbstractNamedDimsArray{T,2}

Derive.interface(::Type{<:AbstractNamedDimsArray}) = NamedDimsArrayInterface()

# Output the dimension names.
nameddimsindices(a::AbstractArray) = throw(MethodError(nameddimsindices, Tuple{typeof(a)}))
# Unwrapping the names
Base.parent(a::AbstractNamedDimsArray) = throw(MethodError(parent, Tuple{typeof(a)}))

nameddimsindices(a::AbstractArray, dim::Int) = nameddimsindices(a)[dim]

function dimnames(a::AbstractNamedDimsArray)
  return name.(nameddimsindices(a))
end
function dimnames(a::AbstractNamedDimsArray, dim::Int)
  return dimnames(a)[dim]
end

function dim(a::AbstractArray, n)
  dimname = to_dimname(a, n)
  return findfirst(==(dimname), nameddimsindices(a))
end
dims(a::AbstractArray, ns) = map(n -> dim(a, n), ns)

dimname_isequal(x) = Base.Fix1(dimname_isequal, x)
dimname_isequal(x, y) = isequal(x, y)

dimname_isequal(r1::AbstractNamedArray, r2::AbstractNamedArray) = isequal(r1, r2)
dimname_isequal(r1::AbstractNamedArray, r2) = name(r1) == r2
dimname_isequal(r1, r2::AbstractNamedArray) = r1 == name(r2)

dimname_isequal(r1::AbstractNamedArray, r2::Name) = name(r1) == name(r2)
dimname_isequal(r1::Name, r2::AbstractNamedArray) = name(r1) == name(r2)

dimname_isequal(r1::AbstractNamedUnitRange, r2::AbstractNamedUnitRange) = isequal(r1, r2)
dimname_isequal(r1::AbstractNamedUnitRange, r2) = name(r1) == r2
dimname_isequal(r1, r2::AbstractNamedUnitRange) = r1 == name(r2)

dimname_isequal(r1::AbstractNamedUnitRange, r2::Name) = name(r1) == name(r2)
dimname_isequal(r1::Name, r2::AbstractNamedUnitRange) = name(r1) == name(r2)

function to_nameddimsindices(a::AbstractArray, dims)
  return to_nameddimsindices(a, axes(a), dims)
end
function to_nameddimsindices(a::AbstractArray, axes, dims)
  return map((axis, dim) -> to_dimname(a, axis, dim), axes, dims)
end
function to_dimname(a::AbstractArray, axis, dim::AbstractNamedArray)
  # TODO: Check `axis` and `dim` have the same shape?
  return dim
end
function to_dimname(a::AbstractArray, axis, dim::AbstractNamedUnitRange)
  # TODO: Check `axis` and `dim` have the same shape?
  return dim
end
# This is the case where just the name of the axis
# was specified without a range, like:
# ```julia
# a = randn(named(2, "i"), named(2, "j"))
# aligndims(a, ("i", "j"))
# ```
function to_dimname(a::AbstractArray, axis, dim)
  return named(axis, dim)
end
function to_dimname(a::AbstractArray, axis, dim::Name)
  return to_dimname(a, axis, name(dim))
end

function to_dimname(a::AbstractNamedDimsArray, dimname)
  dim = findfirst(dimname_isequal(dimname), nameddimsindices(a))
  return to_dimname(a, axes(a, dim), dimname)
end

function to_dimname(a::AbstractNamedDimsArray, axis, dim::AbstractNamedArray)
  return dim
end
function to_dimname(a::AbstractNamedDimsArray, axis, dim::AbstractNamedUnitRange)
  return dim
end
function to_dimname(a::AbstractNamedDimsArray, axis, dim)
  return named(dename(axis), dim)
end
function to_dimname(a::AbstractNamedDimsArray, axis, dim::Name)
  return to_dimname(a, axis, name(dim))
end

function to_nameddimsindices(a::AbstractNamedDimsArray, dims)
  return map(dim -> to_dimname(a, dim), dims)
end

# Unwrapping the names (`NamedDimsArrays.jl` interface).
# TODO: Use `IsNamed` trait?
dename(a::AbstractNamedDimsArray) = parent(a)
function dename(a::AbstractNamedDimsArray, nameddimsindices)
  return dename(aligndims(a, nameddimsindices))
end
function denamed(a::AbstractNamedDimsArray, nameddimsindices)
  return dename(aligneddims(a, nameddimsindices))
end

unname(a::AbstractArray, nameddimsindices) = dename(a, nameddimsindices)
unnamed(a::AbstractArray, nameddimsindices) = denamed(a, nameddimsindices)

isnamed(::Type{<:AbstractNamedDimsArray}) = true

# TODO: Move to `utils.jl` file.
# TODO: Use `Base.indexin`?
function getperm(x, y; isequal=isequal)
  return map(yᵢ -> findfirst(isequal(yᵢ), x), y)
end

# TODO: Move to `utils.jl` file.
function checked_indexin(x, y)
  I = indexin(x, y)
  return something.(I)
end

function checked_indexin(x::Number, y)
  return findfirst(==(x), y)
end

function checked_indexin(x::AbstractUnitRange, y::AbstractUnitRange)
  return findfirst(==(first(x)), y):findfirst(==(last(x)), y)
end

Base.copy(a::AbstractNamedDimsArray) = nameddims(copy(dename(a)), nameddimsindices(a))

const NamedDimsIndices = Union{
  AbstractNamedUnitRange{<:Integer},AbstractNamedArray{<:Integer}
}
const NamedDimsAxis = AbstractNamedUnitRange{
  <:Integer,<:AbstractUnitRange,<:NamedDimsIndices
}

# Generic constructor.
function nameddims(a::AbstractArray, nameddimsindices)
  # TODO: Check the shape of `nameddimsindices` matches the shape of `a`.
  arrtype = mapreduce(nameddimsarraytype, combine_nameddimsarraytype, nameddimsindices)
  return arrtype(a, to_nameddimsindices(a, nameddimsindices))
end

# Can overload this to get custom named dims array wrapper
# depending on the dimension name types, for example
# output an `ITensor` if the dimension names are `IndexName`s.
nameddimsarraytype(nameddim) = nameddimsarraytype(typeof(nameddim))
nameddimsarraytype(nameddimtype::Type) = NamedDimsArray
function nameddimsarraytype(nameddimtype::Type{<:NamedDimsIndices})
  return nameddimsarraytype(nametype(nameddimtype))
end
function combine_nameddimsarraytype(
  ::Type{<:AbstractNamedDimsArray}, ::Type{<:AbstractNamedDimsArray}
)
  return NamedDimsArray
end
combine_nameddimsarraytype(::Type{T}, ::Type{T}) where {T<:AbstractNamedDimsArray} = T

Base.axes(a::AbstractNamedDimsArray) = map(named, axes(dename(a)), nameddimsindices(a))
Base.size(a::AbstractNamedDimsArray) = map(named, size(dename(a)), nameddimsindices(a))

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

Base.axes(a::AbstractNamedDimsArray, dimname::Name) = axes(a, dim(a, dimname))
Base.size(a::AbstractNamedDimsArray, dimname::Name) = size(a, dim(a, dimname))

to_nameddimsaxes(dims) = map(to_nameddimsaxis, dims)
to_nameddimsaxis(ax::NamedDimsAxis) = ax
to_nameddimsaxis(I::NamedDimsIndices) = named(dename(only(axes(I))), I)

function Base.similar(
  a::AbstractArray, elt::Type, inds::Tuple{NamedDimsIndices,Vararg{NamedDimsIndices}}
)
  ax = to_nameddimsaxes(inds)
  return nameddims(similar(unname(a), elt, dename.(ax)), name.(ax))
end

function setnameddimsindices(a::AbstractNamedDimsArray, nameddimsindices)
  return nameddims(dename(a), nameddimsindices)
end
function replacenameddimsindices(f, a::AbstractNamedDimsArray)
  return setnameddimsindices(a, replace(f, nameddimsindices(a)))
end
function replacenameddimsindices(
  a::AbstractNamedDimsArray,
  replacements::Pair{<:AbstractNamedUnitRange,<:AbstractNamedUnitRange}...,
)
  return setnameddimsindices(a, replace(nameddimsindices(a), replacements...))
end
function replacenameddimsindices(a::AbstractNamedDimsArray, replacements::Pair...)
  old_nameddimsindices = to_nameddimsindices(a, first.(replacements))
  new_nameddimsindices = named.(dename.(old_nameddimsindices), last.(replacements))
  return replacenameddimsindices(a, (old_nameddimsindices .=> new_nameddimsindices)...)
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
struct NamedDimsCartesianIndex{N,Index<:Tuple{Vararg{AbstractNamedInteger,N}}} <:
       Base.AbstractCartesianIndex{N}
  I::Index
end
NamedDimsCartesianIndex(I::AbstractNamedInteger...) = NamedDimsCartesianIndex(I)
Base.Tuple(I::NamedDimsCartesianIndex) = I.I
function Base.show(io::IO, I::NamedDimsCartesianIndex)
  print(io, "NamedDimsCartesianIndex")
  show(io, Tuple(I))
  return nothing
end

# Like CartesianIndices but with named dimensions.
struct NamedDimsCartesianIndices{
  N,
  Indices<:Tuple{Vararg{AbstractNamedUnitRange,N}},
  Index<:Tuple{Vararg{AbstractNamedInteger,N}},
} <: AbstractNamedDimsArray{NamedDimsCartesianIndex{N,Index},N}
  indices::Indices
  function NamedDimsCartesianIndices(indices::Tuple{Vararg{AbstractNamedUnitRange}})
    return new{length(indices),typeof(indices),Tuple{eltype.(indices)...}}(indices)
  end
end

Base.eltype(I::NamedDimsCartesianIndices) = eltype(typeof(I))
Base.axes(I::NamedDimsCartesianIndices) = map(only ∘ axes, I.indices)
Base.size(I::NamedDimsCartesianIndices) = length.(I.indices)

function Base.getindex(a::NamedDimsCartesianIndices{N}, I::Vararg{Int,N}) where {N}
  # TODO: Check if `nameddimsindices(a)` is correct here.
  index = map(nameddimsindices(a), I) do r, i
    return r[i]
  end
  return NamedDimsCartesianIndex(index)
end

nameddimsindices(I::NamedDimsCartesianIndices) = name.(I.indices)
function Base.parent(I::NamedDimsCartesianIndices)
  return CartesianIndices(dename.(I.indices))
end

function Base.eachindex(::NamedIndexCartesian, a1::AbstractArray, a_rest::AbstractArray...)
  all(a -> issetequal(nameddimsindices(a1), nameddimsindices(a)), a_rest) ||
    throw(NameMismatch("Dimension name mismatch $(nameddimsindices.((a1, a_rest...)))."))
  # TODO: Check the shapes match.
  return NamedDimsCartesianIndices(axes(a1))
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

# Indexing.

# Scalar indexing

function Base.getindex(a::AbstractNamedDimsArray, I1::Int, Irest::Int...)
  return getindex(dename(a), I1, Irest...)
end
function Base.getindex(a::AbstractNamedDimsArray, I::CartesianIndex)
  return getindex(a, to_indices(a, (I,))...)
end
function Base.getindex(
  a::AbstractNamedDimsArray, I1::AbstractNamedInteger, Irest::AbstractNamedInteger...
)
  I = (I1, Irest...)
  # TODO: Check if this permuation should be inverted.
  perm = getperm(name.(nameddimsindices(a)), name.(I))
  # TODO: Throw a `NameMismatch` error.
  @assert isperm(perm)
  I = map(p -> I[p], perm)
  subinds = map(nameddimsindices(a), I) do dimname, i
    return checked_indexin(dename(i), dename(dimname))
  end
  return getindex(dename(a), subinds...)
end
function Base.getindex(a::AbstractNamedDimsArray, I::NamedDimsCartesianIndex)
  return getindex(a, Tuple(I)...)
end
function Base.getindex(a::AbstractNamedDimsArray, I1::Pair, Irest::Pair...)
  I = (I1, Irest...)
  nameddimsindices = to_nameddimsindices(a, first.(I))
  return getindex(a, map((i, name) -> name[i], last.(I), nameddimsindices)...)
end
function Base.getindex(a::AbstractNamedDimsArray)
  return getindex(dename(a))
end
# Linear indexing.
function Base.getindex(a::AbstractNamedDimsArray, I::Int)
  return getindex(dename(a), I)
end

function Base.setindex!(a::AbstractNamedDimsArray, value, I1::Int, Irest::Int...)
  setindex!(dename(a), value, I1, Irest...)
  return a
end
function Base.setindex!(a::AbstractNamedDimsArray, value, I::CartesianIndex)
  setindex!(a, value, to_indices(a, (I,))...)
  return a
end
function Base.setindex!(
  a::AbstractNamedDimsArray, value, I1::AbstractNamedInteger, Irest::AbstractNamedInteger...
)
  I = (I1, Irest...)
  # TODO: Check if this permuation should be inverted.
  perm = getperm(name.(nameddimsindices(a)), name.(I))
  # TODO: Throw a `NameMismatch` error.
  @assert isperm(perm)
  I = map(p -> I[p], perm)
  subinds = map(nameddimsindices(a), I) do dimname, i
    return checked_indexin(dename(i), dename(dimname))
  end
  return setindex!(dename(a), value, subinds...)
end
function Base.setindex!(a::AbstractNamedDimsArray, value, I::NamedDimsCartesianIndex)
  setindex!(a, value, Tuple(I)...)
  return a
end
function Base.setindex!(a::AbstractNamedDimsArray, value, I1::Pair, Irest::Pair...)
  I = (I1, Irest...)
  nameddimsindices = to_nameddimsindices(a, first.(I))
  setindex!(a, value, map((i, name) -> name[i], last.(I), nameddimsindices)...)
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

function Base.isassigned(a::AbstractNamedDimsArray, I::Int...)
  return isassigned(dename(a), I...)
end

# Slicing

# Like `const ViewIndex = Union{Real,AbstractArray}`.
const NamedViewIndex = Union{AbstractNamedInteger,AbstractNamedUnitRange,AbstractNamedArray}

using ArrayLayouts: ArrayLayouts, MemoryLayout

abstract type AbstractNamedDimsArrayLayout <: MemoryLayout end
struct NamedDimsArrayLayout{ParentLayout} <: AbstractNamedDimsArrayLayout end

function ArrayLayouts.MemoryLayout(arrtype::Type{<:AbstractNamedDimsArray})
  return NamedDimsArrayLayout{typeof(MemoryLayout(parenttype(arrtype)))}()
end

function ArrayLayouts.sub_materialize(::NamedDimsArrayLayout, a, ax)
  return copy(a)
end

function Base.view(a::AbstractArray, I1::NamedViewIndex, Irest::NamedViewIndex...)
  I = (I1, Irest...)
  sub_dims = filter(dim -> I[dim] isa AbstractArray, ntuple(identity, ndims(a)))
  sub_nameddimsindices = map(dim -> I[dim], sub_dims)
  return nameddims(view(a, dename.(I)...), sub_nameddimsindices)
end

function Base.getindex(a::AbstractArray, I1::NamedViewIndex, Irest::NamedViewIndex...)
  return copy(view(a, I1, Irest...))
end

function Base.view(a::AbstractArray, I1::Name, Irest::Name...)
  return nameddims(a, name.((I1, Irest...)))
end

function Base.view(a::AbstractNamedDimsArray, I1::Name, Irest::Name...)
  return view(a, to_nameddimsindices(a, (I1, Irest...))...)
end

function Base.getindex(a::AbstractArray, I1::Name, Irest::Name...)
  return copy(view(a, I1, Irest...))
end

function Base.view(a::AbstractNamedDimsArray, I1::NamedViewIndex, Irest::NamedViewIndex...)
  I = (I1, Irest...)
  # TODO: Check if this permuation should be inverted.
  perm = getperm(name.(nameddimsindices(a)), name.(I))
  # TODO: Throw a `NameMismatch` error.
  @assert isperm(perm)
  I = map(p -> I[p], perm)
  sub_dims = filter(dim -> I[dim] isa AbstractArray, ntuple(identity, ndims(a)))
  sub_nameddimsindices = map(dim -> I[dim], sub_dims)
  subinds = map(nameddimsindices(a), I) do dimname, i
    return checked_indexin(dename(i), dename(dimname))
  end
  return nameddims(view(dename(a), subinds...), sub_nameddimsindices)
end

function Base.getindex(
  a::AbstractNamedDimsArray, I1::NamedViewIndex, Irest::NamedViewIndex...
)
  return copy(view(a, I1, Irest...))
end

# Repeated definition of `Base.ViewIndex`.
const ViewIndex = Union{Real,AbstractArray}

function nameddims_view(a::AbstractArray, I...)
  sub_dims = filter(dim -> !(I[dim] isa Real), ntuple(identity, ndims(a)))
  sub_nameddimsindices = map(dim -> nameddimsindices(a, dim)[I[dim]], sub_dims)
  return nameddims(view(dename(a), I...), sub_nameddimsindices)
end

function Base.view(a::AbstractNamedDimsArray, I::ViewIndex...)
  return nameddims_view(a, I...)
end

function nameddims_getindex(a::AbstractArray, I...)
  return copy(view(a, I...))
end

function Base.getindex(a::AbstractNamedDimsArray, I::ViewIndex...)
  return nameddims_getindex(a, I...)
end

function Base.setindex!(
  a::AbstractNamedDimsArray,
  value::AbstractNamedDimsArray,
  I1::NamedViewIndex,
  Irest::NamedViewIndex...,
)
  view(a, I1, Irest...) .= value
  return a
end
function Base.setindex!(
  a::AbstractNamedDimsArray,
  value::AbstractArray,
  I1::NamedViewIndex,
  Irest::NamedViewIndex...,
)
  I = (I1, Irest...)
  setindex!(a, nameddims(value, I), I...)
  return a
end
function Base.setindex!(
  a::AbstractNamedDimsArray,
  value::AbstractNamedDimsArray,
  I1::ViewIndex,
  Irest::ViewIndex...,
)
  view(a, I1, Irest...) .= value
  return a
end
function Base.setindex!(
  a::AbstractNamedDimsArray, value::AbstractArray, I1::ViewIndex, Irest::ViewIndex...
)
  setindex!(dename(a), value, I1, Irest...)
  return a
end

# Permute/align dimensions

function aligndims(a::AbstractArray, dims)
  new_nameddimsindices = to_nameddimsindices(a, dims)
  # TODO: Check this permutation is correct (it may be the inverse of what we want).
  perm = getperm(nameddimsindices(a), new_nameddimsindices)
  isperm(perm) || throw(
    NameMismatch(
      "Dimension name mismatch $(nameddimsindices(a)), $(new_nameddimsindices)."
    ),
  )
  return nameddims(permutedims(dename(a), perm), new_nameddimsindices)
end

function aligneddims(a::AbstractArray, dims)
  new_nameddimsindices = to_nameddimsindices(a, dims)
  # TODO: Check this permutation is correct (it may be the inverse of what we want).
  perm = getperm(nameddimsindices(a), new_nameddimsindices)
  isperm(perm) || throw(
    NameMismatch(
      "Dimension name mismatch $(nameddimsindices(a)), $(new_nameddimsindices)."
    ),
  )
  return nameddims(PermutedDimsArray(dename(a), perm), new_nameddimsindices)
end

# Convenient constructors

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

default_eltype() = Float64
for (f, f′) in [(:rand, :_rand), (:randn, :_randn)]
  @eval begin
    function Base.$f(
      rng::AbstractRNG,
      elt::Type{<:Number},
      inds::Tuple{NamedDimsIndices,Vararg{NamedDimsIndices}},
    )
      ax = to_nameddimsaxes(inds)
      a = $f′(rng, elt, dename.(ax))
      return nameddims(a, name.(ax))
    end
    function Base.$f(
      rng::AbstractRNG,
      elt::Type{<:Number},
      dims::Tuple{AbstractNamedInteger,Vararg{AbstractNamedInteger}},
    )
      return $f(rng, elt, Base.oneto.(dims))
    end
  end
  for dimtype in [:AbstractNamedInteger, :NamedDimsIndices]
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
  @eval begin
    function Base.$f(
      elt::Type{<:Number}, ax::Tuple{NamedDimsIndices,Vararg{NamedDimsIndices}}
    )
      ax = to_nameddimsaxes(inds)
      a = $f(elt, dename.(ax))
      return nameddims(a, name.(ax))
    end
    function Base.$f(
      elt::Type{<:Number}, dims::Tuple{AbstractNamedInteger,Vararg{AbstractNamedInteger}}
    )
      a = $f(elt, dename.(dims))
      return nameddims(a, Base.oneto.(dims))
    end
  end
  for dimtype in [:AbstractNamedInteger, :NamedDimsIndices]
    @eval begin
      function Base.$f(elt::Type{<:Number}, dim1::$dimtype, dims::Vararg{$dimtype})
        return $f(elt, (dim1, dims...))
      end
      Base.$f(dims::Tuple{$dimtype,Vararg{$dimtype}}) = $f(default_eltype(), dims)
      Base.$f(dim1::$dimtype, dims::Vararg{$dimtype}) = $f((dim1, dims...))
    end
  end
end
@eval begin
  function Base.fill(value, inds::Tuple{NamedDimsIndices,Vararg{NamedDimsIndices}})
    ax = to_nameddimsaxes(inds)
    a = fill(value, dename.(ax))
    return nameddims(a, name.(ax))
  end
  function Base.fill(value, dims::Tuple{AbstractNamedInteger,Vararg{AbstractNamedInteger}})
    a = fill(value, dename.(dims))
    return nameddims(a, Base.oneto.(dims))
  end
end
for dimtype in [:AbstractNamedInteger, :NamedDimsIndices]
  @eval begin
    function Base.fill(value, dim1::$dimtype, dims::Vararg{$dimtype})
      return fill(value, (dim1, dims...))
    end
  end
end

# Broadcasting

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
# TODO: Make a version that gets the nameddimsindices from `m`.
function denamed(m::Mapped, nameddimsindices)
  return mapped(m.f, map(arg -> denamed(arg, nameddimsindices), m.args)...)
end

function Base.similar(bc::Broadcasted{<:AbstractNamedDimsArrayStyle}, elt::Type, ax::Tuple)
  nameddimsindices = name.(ax)
  m′ = denamed(Mapped(bc), nameddimsindices)
  return nameddims(similar(m′, elt, dename.(ax)), nameddimsindices)
end

function Base.copyto!(
  dest::AbstractArray{<:Any,N}, bc::Broadcasted{<:AbstractNamedDimsArrayStyle{N}}
) where {N}
  return copyto!(dest, Mapped(bc))
end

function Base.map!(f, a_dest::AbstractNamedDimsArray, a_srcs::AbstractNamedDimsArray...)
  a′_dest = dename(a_dest)
  # TODO: Use `denamed` to do the permutations lazily.
  a′_srcs = map(a_src -> dename(a_src, nameddimsindices(a_dest)), a_srcs)
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

# Printing

function Base.show(io::IO, mime::MIME"text/plain", a::AbstractNamedDimsArray)
  summary(io, a)
  println(io)
  show(io, mime, dename(a))
  return nothing
end

function Base.show(io::IO, a::AbstractNamedDimsArray)
  print(io, "nameddims(")
  show(io, dename(a))
  print(io, ", ", nameddimsindices(a), ")")
  return nothing
end
