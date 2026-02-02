import FunctionImplementations as FI
using TypeParameterAccessors: unspecify_type_parameters

# Some of the interface is inspired by:
# https://github.com/ITensor/ITensors.jl
# https://github.com/invenia/NamedDims.jl
# https://github.com/mcabbott/NamedPlus.jl
# https://pytorch.org/docs/stable/named_tensor.html

abstract type AbstractNamedDimsArrayImplementationStyle <:
FI.AbstractArrayImplementationStyle end

struct NamedDimsArrayImplementationStyle <: AbstractNamedDimsArrayImplementationStyle end

abstract type AbstractNamedDimsArray{T, N} <: AbstractArray{T, N} end

const AbstractNamedDimsVector{T} = AbstractNamedDimsArray{T, 1}
const AbstractNamedDimsMatrix{T} = AbstractNamedDimsArray{T, 2}

function FI.ImplementationStyle(type::Type{<:AbstractNamedDimsArray})
    return NamedDimsArrayImplementationStyle()
end

const NamedDimsIndices = Union{
    AbstractNamedUnitRange{<:Integer}, AbstractNamedArray{<:Integer},
}
const NamedDimsAxis = AbstractNamedUnitRange{
    <:Integer, <:AbstractUnitRange, <:NamedDimsIndices,
}

dimnames(a::AbstractNamedDimsArray) = throw(MethodError(dimnames, a))
function dimnames(a::AbstractNamedDimsArray, dim::Int)
    return dimnames(a)[dim]
end

# Unwrapping the names (`NamedDimsArrays.jl` interface).
# TODO: Use `IsNamed` trait?
denamed(a::AbstractNamedDimsArray) = throw(MethodError(denamed, a))
denamed(a::AbstractNamedDimsArray, inds) = denamed(aligneddims(a, inds))
dename(a::AbstractNamedDimsArray, inds) = denamed(aligndims(a, inds))

# Output the named axes/indices of the named dims array.
inds(a::AbstractArray) = LittleSet(named.(axes(denamed(a)), dimnames(a)))
inds(a::AbstractArray, dim::Int) = inds(a)[dim]

isnamed(::Type{<:AbstractNamedDimsArray}) = true

function dim(a::AbstractArray, n)
    return findfirst(==(name(n)), dimnames(a))
end
dims(a::AbstractArray, ns) = Base.Fix1(dim, a).(ns)

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

function to_inds(a::AbstractArray, dims)
    is = Base.Fix1(dim, a).(name.(dims))
    return Base.Fix1(inds, a).(is)
end

# Generic construction of named dims arrays.

"""
    nameddims(a::AbstractArray, inds)

Construct a named dimensions array from an denamed array `a` and named dimensions `inds`.
"""
function nameddims(a::AbstractArray, inds)
    return nameddimsconstructor(a, inds)(a, inds)
end

#=
    nameddimsof(a::AbstractNamedDimsArray, b::AbstractArray)

Construct a named dimensions array with the dimension names of `a`
and based on the type of `a` but with the data from `b`.
=#
function nameddimsof(a::AbstractNamedDimsArray, b::AbstractArray)
    return nameddimsconstructorof(a)(b, dimnames(a))
end

# Interface inspired by
# [ConstructionBase.constructorof](https://github.com/JuliaObjects/ConstructionBase.jl).
nameddimsconstructorof(a::AbstractNamedDimsArray) = nameddimsconstructorof(typeof(a))
function nameddimsconstructorof(type::Type{<:AbstractNamedDimsArray})
    return unspecify_type_parameters(type)
end

# Output a constructor for a named dims array (that should accept and denamed array and
# a set of named dimensions/axes/indices) based on the dimension names.
function nameddimsconstructor(a::AbstractArray, dims)
    dimnames = name.(dims)
    isempty(dimnames) && return NamedDimsArray
    return mapreduce(nameddimsconstructor, combine_nameddimsconstructors, dimnames)
end

nameddimsconstructor(nameddim) = nameddimsconstructor(typeof(nameddim))
# Can overload this to get custom named dims array wrapper
# depending on the dimension name types, for example
# output an `ITensor` if the dimension names are `IndexName`s.
nameddimsconstructor(nameddimtype::Type) = NamedDimsArray
function nameddimsconstructor(nameddimtype::Type{<:NamedDimsIndices})
    return nameddimsconstructor(nametype(nameddimtype))
end
function combine_nameddimsconstructors(
        ::Type{<:AbstractNamedDimsArray}, ::Type{<:AbstractNamedDimsArray}
    )
    return NamedDimsArray
end
combine_nameddimsconstructors(::Type{T}, ::Type{T}) where {T <: AbstractNamedDimsArray} = T

# TODO: Move to `utils.jl` file.
# TODO: Use `Base.indexin`?
function getperm(x, y; isequal = isequal)
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

Base.copy(a::AbstractNamedDimsArray) = nameddimsof(a, copy(denamed(a)))

function Base.copyto!(a_dest::AbstractNamedDimsArray, a_src::AbstractNamedDimsArray)
    a′_dest = denamed(a_dest)
    # TODO: Use `denamed` to do the permutations lazily.
    a′_src = dename(a_src, inds(a_dest))
    copyto!(a′_dest, a′_src)
    return a_dest
end

# Conversion

# Copied from `Base` (defined in abstractarray.jl).
@noinline function _checkaxs(axd, axs)
    axd == axs || throw(DimensionMismatch("axes must agree, got $axd and $axs"))
    return nothing
end
function copyto_axcheck!(dest, src)
    _checkaxs(axes(dest), axes(src))
    copyto!(dest, src)
    return dest
end

# These are defined since the Base versions assume the eltype and ndims are known
# at compile time, which isn't true for ITensors.
Base.Array(a::AbstractNamedDimsArray) = Array(denamed(a))
Base.Array{T}(a::AbstractNamedDimsArray) where {T} = Array{T}(denamed(a))
Base.Array{T, N}(a::AbstractNamedDimsArray) where {T, N} = Array{T, N}(denamed(a))
Base.AbstractArray{T}(a::AbstractNamedDimsArray) where {T} = AbstractArray{T, ndims(a)}(a)
function Base.AbstractArray{T, N}(a::AbstractNamedDimsArray) where {T, N}
    dest = similar(a, T)
    copyto_axcheck!(denamed(dest), denamed(a))
    return dest
end

function Base.axes(a::AbstractNamedDimsArray)
    return inds(a)
end
function Base.size(a::AbstractNamedDimsArray)
    return length.(axes(a))
end

function Base.length(a::AbstractNamedDimsArray)
    return prod(size(a); init = 1)
end

# Circumvent issue when ndims isn't known at compile time.
Base.axes(a::AbstractNamedDimsArray, d) = axes(a)[d]

# Circumvent issue when ndims isn't known at compile time.
Base.size(a::AbstractNamedDimsArray, d) = size(a)[d]

# Circumvent issue when ndims isn't known at compile time.
Base.ndims(a::AbstractNamedDimsArray) = ndims(denamed(a))

# Circumvent issue when eltype isn't known at compile time.
Base.eltype(a::AbstractNamedDimsArray) = eltype(denamed(a))

using VectorInterface: VectorInterface, scalartype
# Circumvent issue when eltype isn't known at compile time.
VectorInterface.scalartype(a::AbstractNamedDimsArray) = scalartype(denamed(a))

Base.axes(a::AbstractNamedDimsArray, dimname::Name) = axes(a, dim(a, dimname))
Base.size(a::AbstractNamedDimsArray, dimname::Name) = size(a, dim(a, dimname))

function similar_nameddims(a::AbstractNamedDimsArray, elt::Type, ax)
    return nameddimsconstructorof(a)(similar(denamed(a), elt, denamed.(Tuple(ax))), name.(ax))
end
function similar_nameddims(a::AbstractArray, elt::Type, ax)
    return nameddims(similar(a, elt, denamed.(Tuple(ax))), name.(ax))
end

# Base.similar gets the eltype at compile time.
Base.similar(a::AbstractNamedDimsArray) = similar(a, eltype(a))
function Base.similar(a::AbstractNamedDimsArray, elt::Type)
    return similar_nameddims(a, elt)
end
function similar_nameddims(a::AbstractNamedDimsArray, elt::Type)
    return nameddimsof(a, similar(denamed(a), elt))
end

# This is defined explicitly since the Base version expects the eltype
# to be known at compile time, which isn't true for ITensors.
function Base.similar(
        a::AbstractArray, inds::Tuple{NamedDimsIndices, Vararg{NamedDimsIndices}}
    )
    return similar(a, eltype(a), inds)
end

function Base.similar(
        a::AbstractArray, elt::Type, inds::Tuple{NamedDimsIndices, Vararg{NamedDimsIndices}}
    )
    return similar_nameddims(a, elt, inds)
end

function Base.similar(a::AbstractArray, inds::LittleSet)
    return similar_nameddims(a, eltype(a), inds)
end

function Base.similar(a::AbstractArray, elt::Type, inds::LittleSet)
    return similar_nameddims(a, elt, inds)
end

function setinds(a::AbstractNamedDimsArray, inds)
    return nameddimsconstructorof(a)(denamed(a), inds)
end

function replacedimnames(a::AbstractNamedDimsArray, replacements::Pair...)
    new_dimnames = replace(dimnames(a), replacements...)
    return nameddims(denamed(a), new_dimnames)
end
function replacedimnames(f, a::AbstractNamedDimsArray)
    new_dimnames = replace(f, dimnames(a))
    return nameddims(denamed(a), new_dimnames)
end
mapdimnames(f, a::AbstractNamedDimsArray) = replacedimnames(f, a)

function replaceinds(a::AbstractNamedDimsArray, replacements::Pair...)
    new_inds = replace(inds(a), replacements...)
    return denamed(a)[new_inds...]
end
function replaceinds(f, a::AbstractNamedDimsArray)
    new_inds = replace(f, inds(a))
    return denamed(a)[new_inds...]
end
mapinds(f, a::AbstractNamedDimsArray) = replaceinds(f, a)

# `Base.isempty(a::AbstractArray)` is defined as `length(a) == 0`,
# which involves comparing a named integer to an denamed integer
# which isn't well defined.
Base.isempty(a::AbstractNamedDimsArray) = isempty(denamed(a))

# Define this on objects rather than types in case the wrapper type
# isn't known at compile time, like for the ITensor type.
Base.IndexStyle(a::AbstractNamedDimsArray) = IndexStyle(denamed(a))
Base.eachindex(a::AbstractNamedDimsArray) = eachindex(denamed(a))

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
## TODO: FIXME ## Delete in favor of using `LittleSet`.
struct NamedDimsCartesianIndex{N, Index <: Tuple{Vararg{AbstractNamedInteger, N}}} <:
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
## TODO: FIXME ## Generalize AbstractNamedUnitRange constraint (for example
# to NamedDimsIndices).
struct NamedDimsCartesianIndices{
        N,
        Indices <: Tuple{Vararg{AbstractNamedUnitRange, N}},
        Index <: Tuple{Vararg{AbstractNamedInteger, N}},
    } <: AbstractNamedDimsArray{NamedDimsCartesianIndex{N, Index}, N}
    indices::Indices
    function NamedDimsCartesianIndices(indices::Tuple{Vararg{AbstractNamedUnitRange}})
        return new{length(indices), typeof(indices), Tuple{eltype.(indices)...}}(indices)
    end
end
function NamedDimsCartesianIndices(indices::LittleSet)
    return NamedDimsCartesianIndices(Tuple(indices))
end

Base.eltype(I::NamedDimsCartesianIndices) = eltype(typeof(I))
Base.axes(I::NamedDimsCartesianIndices) = (only ∘ axes ∘ denamed).(I.indices)
Base.size(I::NamedDimsCartesianIndices) = (length ∘ denamed).(I.indices)

function Base.getindex(a::NamedDimsCartesianIndices{N}, I::Vararg{Int, N}) where {N}
    index = map(a.indices, I) do r, i
        return r[i]
    end
    ## TODO: FIXME ## Output a `LittleSet` instead.
    return NamedDimsCartesianIndex(index)
end

function denamed(I::NamedDimsCartesianIndices)
    return CartesianIndices(denamed.(I.indices))
end

function Base.eachindex(::NamedIndexCartesian, a1::AbstractArray, a_rest::AbstractArray...)
    all(a -> issetequal(inds(a1), inds(a)), a_rest) ||
        throw(NameMismatch("Dimension name mismatch $(inds.((a1, a_rest...)))."))
    # TODO: Check the shapes match.
    return NamedDimsCartesianIndices(inds(a1))
end

# Base version ignores dimension names.
# TODO: Use `mapreduce(isequal, &&, a1, a2)`?
function Base.isequal(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
    (inds(a1) == inds(a2)) || return false
    return isequal(denamed(a1), denamed(a2, inds(a1)))
end

# Base version ignores dimension names.
# TODO: Use `mapreduce(==, &&, a1, a2)`?
# TODO: Handle `missing` values properly.
function Base.:(==)(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
    (inds(a1) == inds(a2)) || return false
    return denamed(a1) == denamed(a2, inds(a1))
end

# Generalization of `Base.sort` to Tuples for Julia v1.10 compatibility.
# TODO: Remove when we drop support for Julia v1.10.
_sort(x; kwargs...) = sort(x; kwargs...)
_sort(x::NTuple{N}; kwargs...) where {N} = NTuple{N}(sort(collect(x); kwargs...))

function Base.hash(a::AbstractNamedDimsArray, h::UInt64)
    h = hash(:NamedDimsArray, h)
    a′ = aligneddims(a, _sort(dimnames(a)))
    h = hash(denamed(a′), h)
    for i in inds(a′)
        h = hash(i, h)
    end
    return h
end

# Indexing.

# Scalar indexing

Base.firstindex(a::AbstractNamedDimsArray) = firstindex(denamed(a))
Base.lastindex(a::AbstractNamedDimsArray) = lastindex(denamed(a))

function Base.firstindex(a::AbstractNamedDimsArray, d)
    return FirstIndex(a, d)
end

function Base.lastindex(a::AbstractNamedDimsArray, d)
    return LastIndex(a, d)
end

# Redefine generic definition which expects `axes(a)` to
# return a Tuple.
function Base.to_indices(a::AbstractNamedDimsArray, I::Tuple)
    return to_indices(a, Tuple(axes(a)), I)
end
# Fix ambiguity error with Base.
function Base.to_indices(a::AbstractNamedDimsArray, I::Tuple{Union{Integer, CartesianIndex}})
    return to_indices(a, Tuple(axes(a)), I)
end
function Base.checkbounds(::Type{Bool}, a::AbstractNamedDimsArray, I::Int...)
    return checkbounds(Bool, denamed(a), I...)
end

function Base.to_indices(
        a::AbstractNamedDimsArray, I::Tuple{AbstractNamedInteger, Vararg{AbstractNamedInteger}}
    )
    perm = getperm(name.(I), dimnames(a))
    # TODO: Throw a `NameMismatch` error.
    @assert isperm(perm)
    I = map(p -> I[p], perm)
    return map(inds(a), I) do dimname, i
        return checked_indexin(denamed(i), denamed(dimname))
    end
end
function Base.to_indices(
        a::AbstractNamedDimsArray, I::Tuple{Pair{<:Any, Int}, Vararg{Pair{<:Any, Int}}}
    )
    inds = to_inds(a, first.(I))
    return to_indices(a, map((i, name) -> name[i], last.(I), inds))
end
function Base.to_indices(a::AbstractNamedDimsArray, I::Tuple{Pair, Vararg{Pair}})
    inds = to_inds(a, first.(I))
    return map((i, name) -> name[i], last.(I), inds)
    return to_indices(a, named.(last.(I), first.(I)))
end

function Base.to_indices(a::AbstractNamedDimsArray, I::Tuple{NamedDimsCartesianIndex})
    return to_indices(a, Tuple(only(I)))
end

function Base.getindex(a::AbstractNamedDimsArray, I...)
    return getindex(a, to_indices(a, I)...)
end

function Base.getindex(a::AbstractNamedDimsArray, I1::Int, Irest::Int...)
    return getindex(denamed(a), I1, Irest...)
end
function Base.getindex(
        a::AbstractNamedDimsArray, I1::AbstractNamedInteger, Irest::AbstractNamedInteger...
    )
    return getindex(a, to_indices(a, (I1, Irest...))...)
end
function Base.getindex(a::AbstractNamedDimsArray)
    return getindex(denamed(a))
end
# Linear indexing.
function Base.getindex(a::AbstractNamedDimsArray, I::Int)
    return getindex(denamed(a), I)
end

function Base.setindex!(a::AbstractNamedDimsArray, value, I1::Int, Irest::Int...)
    setindex!(denamed(a), value, I1, Irest...)
    return a
end
function Base.setindex!(a::AbstractNamedDimsArray, value, I::CartesianIndex)
    setindex!(a, value, to_indices(a, (I,))...)
    return a
end

function Base.setindex!(
        a::AbstractNamedDimsArray, value, I1::AbstractNamedInteger, Irest::AbstractNamedInteger...
    )
    setindex!(a, value, to_indices(a, (I1, Irest...))...)
    return a
end
function Base.setindex!(a::AbstractNamedDimsArray, value, I::NamedDimsCartesianIndex)
    setindex!(a, value, to_indices(a, (I,))...)
    return a
end
function Base.setindex!(a::AbstractNamedDimsArray, value, I1::Pair, Irest::Pair...)
    setindex!(a, value, to_indices(a, (I1, Irest...))...)
    return a
end
function Base.setindex!(a::AbstractNamedDimsArray, value)
    setindex!(denamed(a), value)
    return a
end
# Linear indexing.
function Base.setindex!(a::AbstractNamedDimsArray, value, I::Int)
    setindex!(denamed(a), value, I)
    return a
end

function Base.isassigned(a::AbstractNamedDimsArray, I::Int...)
    return isassigned(denamed(a), I...)
end

# Slicing

# Like `const ViewIndex = Union{Real,AbstractArray}`.
const NamedViewIndex = Union{AbstractNamedInteger, AbstractNamedUnitRange, AbstractNamedArray}

using ArrayLayouts: ArrayLayouts, MemoryLayout

abstract type AbstractNamedDimsArrayLayout <: MemoryLayout end
struct NamedDimsArrayLayout{ParentLayout} <: AbstractNamedDimsArrayLayout end

function ArrayLayouts.MemoryLayout(arrtype::Type{<:AbstractNamedDimsArray})
    return NamedDimsArrayLayout{typeof(MemoryLayout(parenttype(arrtype)))}()
end

function ArrayLayouts.sub_materialize(::NamedDimsArrayLayout, a, ax)
    return copy(a)
end

# TODO: Should this be a view?
function Base.getindex(a::AbstractArray, I1::Name, Irest::Name...)
    return copy(view(a, I1, Irest...))
end
function Base.view(a::AbstractArray, I1::Name, Irest::Name...)
    return nameddims(a, name.((I1, Irest...)))
end

function Base.getindex(a::AbstractArray, I1::NamedViewIndex, Irest::NamedViewIndex...)
    return copy(view(a, I1, Irest...))
end
# Disambiguate from `Base.getindex(A::Array, I::AbstractUnitRange{<:Integer})`.
function Base.getindex(a::Array, I1::AbstractNamedUnitRange{<:Integer})
    return copy(view(a, I1))
end
function Base.view(a::AbstractArray, I1::NamedViewIndex, Irest::NamedViewIndex...)
    I = (I1, Irest...)
    return nameddims(view(a, denamed.(I)...), name.(I))
end

# TODO: Should this be a view?
function Base.getindex(a::AbstractNamedDimsArray, I1::Name, Irest::Name...)
    return copy(view(a, I1, Irest...))
end
function Base.view(a::AbstractNamedDimsArray, I1::Name, Irest::Name...)
    issetequal(dimnames(a), name.((I1, Irest...))) ||
        throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(name.((I1, Irest...)))."
        ),
    )
    return a
end

function Base.getindex(a::AbstractNamedDimsArray, I1::Pair, Irest::Pair...)
    return getindex(a, to_indices(a, (I1, Irest...))...)
end
function Base.view(a::AbstractNamedDimsArray, I1::Pair, Irest::Pair...)
    I = (I1, Irest...)
    inds = to_inds(a, first.(I))
    return view(a, map((i, name) -> name[i], last.(I), inds)...)
end

function Base.getindex(
        a::AbstractNamedDimsArray, I1::NamedViewIndex, Irest::NamedViewIndex...
    )
    return copy(view(a, I1, Irest...))
end
function Base.view(a::AbstractNamedDimsArray, I1::NamedViewIndex, Irest::NamedViewIndex...)
    I = (I1, Irest...)
    perm = getperm(name.(I), dimnames(a))
    isperm(perm) || throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(name.(I))."
        ),
    )
    Ip = map(p -> denamed(I[p]), perm)
    return view_nameddims(a, Ip...)
end

# Repeated definition of `Base.ViewIndex`.
const ViewIndex = Union{Real, AbstractArray}

# Slicing with unnamed indices, such as:
# a = NamedDimsArray(rand(3,4), (:x, :y))
# b = view(a, 1:2, 2)
function view_nameddims(a::AbstractNamedDimsArray, I...)
    nonscalar_dims = filter(dim -> I[dim] isa AbstractArray, ntuple(identity, ndims(a)))
    nonscalar_dimnames = map(dim -> dimnames(a, dim), nonscalar_dims)
    return nameddimsconstructorof(a)(view(denamed(a), I...), nonscalar_dimnames)
end

function Base.view(a::AbstractNamedDimsArray, I::ViewIndex...)
    return view_nameddims(a, I...)
end

function getindex_nameddims(a::AbstractArray, I...)
    return copy(view(a, I...))
end

function Base.getindex(a::AbstractNamedDimsArray, I::ViewIndex...)
    return getindex_nameddims(a, I...)
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
    a[I...] = nameddimsconstructorof(a)(value, I)
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
    setindex!(denamed(a), value, I1, Irest...)
    return a
end

# Permute/align dimensions

function aligndims(a::AbstractArray, dims)
    new_dimnames = name.(dims)
    perm = Tuple(getperm(dimnames(a), new_dimnames))
    isperm(perm) || throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(new_dimnames)."
        ),
    )
    return nameddimsconstructorof(a)(permutedims(denamed(a), perm), new_dimnames)
end

function aligneddims(a::AbstractArray, dims)
    new_dimnames = name.(dims)
    perm = getperm(dimnames(a), new_dimnames)
    isperm(perm) || throw(
        NameMismatch(
            "Dimension name mismatch $(dimnames(a)), $(new_dimnames)."
        ),
    )
    return nameddimsconstructorof(a)(
        FI.permuteddims(denamed(a), perm), new_dimnames
    )
end

# Convenient constructors

using Random: Random, AbstractRNG

# Like `Base.rand` but supports axes, not just size.
# TODO: Come up with a better name for this.
_rand(args...) = Base.rand(args...)
function _rand(
        rng::AbstractRNG, elt::Type, dims::Tuple{Base.OneTo{Int}, Vararg{Base.OneTo{Int}}}
    )
    return Base.rand(rng, elt, length.(dims))
end

# Like `Base.randn` but supports axes, not just size.
# TODO: Come up with a better name for this.
_randn(args...) = Base.randn(args...)
function _randn(
        rng::AbstractRNG, elt::Type, dims::Tuple{Base.OneTo{Int}, Vararg{Base.OneTo{Int}}}
    )
    return Base.randn(rng, elt, length.(dims))
end

default_eltype() = Float64
for (f, f′) in [(:rand, :_rand), (:randn, :_randn)]
    @eval begin
        function Base.$f(
                rng::AbstractRNG,
                elt::Type{<:Number},
                ax::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}},
            )
            a = $f′(rng, elt, denamed.(ax))
            return a[Name.(name.(ax))...]
        end
        function Base.$f(
                rng::AbstractRNG,
                elt::Type{<:Number},
                dims::Tuple{AbstractNamedInteger, Vararg{AbstractNamedInteger}},
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
            Base.$f(elt::Type{<:Number}, dims::Tuple{$dimtype, Vararg{$dimtype}}) = $f(
                Random.default_rng(), elt, dims
            )
            Base.$f(elt::Type{<:Number}, dim1::$dimtype, dims::Vararg{$dimtype}) = $f(
                elt, (dim1, dims...)
            )
            Base.$f(dims::Tuple{$dimtype, Vararg{$dimtype}}) = $f(default_eltype(), dims)
            Base.$f(dim1::$dimtype, dims::Vararg{$dimtype}) = $f((dim1, dims...))
        end
    end
end
for f in [:zeros, :ones]
    @eval begin
        # TODO: FIXME: Combine these two definitions into a single one in a loop over `@eval`.
        function Base.$f(
                elt::Type{<:Number}, ax::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
            )
            a = $f(elt, denamed.(ax))
            return a[Name.(name.(ax))...]
        end
        # TODO: FIXME: Combine these two definitions into a single one in a loop over `@eval`.
        function Base.$f(
                elt::Type{<:Number}, ax::Tuple{AbstractNamedInteger, Vararg{AbstractNamedInteger}}
            )
            a = $f(elt, denamed.(ax))
            return a[Name.(name.(ax))...]
        end
    end
    for dimtype in [:AbstractNamedInteger, :AbstractNamedUnitRange]
        @eval begin
            function Base.$f(elt::Type{<:Number}, dim1::$dimtype, dims::Vararg{$dimtype})
                return $f(elt, (dim1, dims...))
            end
            Base.$f(dims::Tuple{$dimtype, Vararg{$dimtype}}) = $f(default_eltype(), dims)
            Base.$f(dim1::$dimtype, dims::Vararg{$dimtype}) = $f((dim1, dims...))
        end
    end
end
@eval begin
    # TODO: FIXME: Combine these two definitions into a single one in a loop over `@eval`.
    function Base.fill(value, ax::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}})
        a = fill(value, denamed.(ax))
        return a[Name.(name.(ax))...]
    end
    # TODO: FIXME: Combine these two definitions into a single one in a loop over `@eval`.
    function Base.fill(value, dims::Tuple{AbstractNamedInteger, Vararg{AbstractNamedInteger}})
        a = fill(value, denamed.(dims))
        return a[Name.(name.(dims))...]
    end
end
for dimtype in [:AbstractNamedInteger, :AbstractNamedUnitRange]
    @eval begin
        function Base.fill(value, dim1::$dimtype, dims::Vararg{$dimtype})
            return fill(value, (dim1, dims...))
        end
    end
end

function Base.map!(f, a_dest::AbstractNamedDimsArray, a_srcs::AbstractNamedDimsArray...)
    a′_dest = denamed(a_dest)
    # TODO: Use `denamed` to do the permutations lazily.
    # TODO: Define `dename[d](dimnames) = Base.Fix1(dename[d], dimnames)` and use it here?
    a′_srcs = Base.Fix2(dename, dimnames(a_dest)).(a_srcs)
    map!(f, a′_dest, a′_srcs...)
    return a_dest
end

function Base.map(f, a_srcs::AbstractNamedDimsArray...)
    # copy(mapped(f, a_srcs...))
    return f.(a_srcs...)
end

function Base.mapreduce(f, op, a::AbstractNamedDimsArray; kwargs...)
    return mapreduce(f, op, denamed(a); kwargs...)
end

using LinearAlgebra: LinearAlgebra, norm
function LinearAlgebra.norm(a::AbstractNamedDimsArray; kwargs...)
    return norm(denamed(a); kwargs...)
end

# Printing

# Copy of `Base.dims2string` defined in `show.jl`.
function dims_to_string(d)
    isempty(d) && return "0-dimensional"
    length(d) == 1 && return "$(d[1])-element"
    return join(map(string, d), '×')
end

using TypeParameterAccessors: type_parameters, unspecify_type_parameters
function concretetype_to_string_truncated(type::Type; param_truncation_length = typemax(Int))
    isconcretetype(type) || throw(ArgumentError("Type must be concrete."))
    alias = Base.make_typealias(type)
    base_type, params = if isnothing(alias)
        unspecify_type_parameters(type), type_parameters(type)
    else
        base_type_globalref, params_svec = alias
        base_type_globalref.name, params_svec
    end
    str = string(base_type)
    if isempty(params)
        return str
    end
    str *= '{'
    param_strings = map(params) do param
        param_string = string(param)
        if length(param_string) > param_truncation_length
            return "…"
        end
        return param_string
    end
    str *= join(param_strings, ", ")
    str *= '}'
    return str
end

function Base.summary(io::IO, a::AbstractNamedDimsArray)
    print(io, dims_to_string(inds(a)))
    print(io, ' ')
    print(io, concretetype_to_string_truncated(typeof(a); param_truncation_length = 40))
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", a::AbstractNamedDimsArray)
    summary(io, a)
    println(io, ":")
    show(io, mime, denamed(a))
    return nothing
end

function Base.show(io::IO, a::AbstractNamedDimsArray)
    show(io, denamed(a))
    print(io, "[", join(inds(a), ", "), "]")
    return nothing
end
