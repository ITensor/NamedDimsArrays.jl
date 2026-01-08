using TypeParameterAccessors: unspecify_type_parameters

abstract type AbstractNamedUnitRange{T, Value <: AbstractUnitRange, Name} <:
AbstractUnitRange{T} end

# Minimal interface.
denamed(r::AbstractNamedUnitRange) = throw(MethodError(denamed, Tuple{typeof(r)}))
name(r::AbstractNamedUnitRange) = throw(MethodError(name, Tuple{typeof(r)}))

# This can be customized to output different named integer types,
# such as `namedunitrange(r::AbstractUnitRange, name::IndexName) = Index(r, name)`.
namedunitrange(r::AbstractUnitRange, name) = NamedUnitRange(r, name)

# Shorthand.
named(r::AbstractUnitRange, name) = namedunitrange(r, name)

# Derived interface.
# TODO: Use `Accessors.@set`?
setname(r::AbstractNamedUnitRange, name) = named(denamed(r), name)

# TODO: Use `TypeParameterAccessors`.
denamedtype(::Type{<:AbstractNamedUnitRange{<:Any, Value}}) where {Value} = Value
nametype(::Type{<:AbstractNamedUnitRange{<:Any, <:Any, Name}}) where {Name} = Name

# Traits.
isnamed(::Type{<:AbstractNamedUnitRange}) = true

# TODO: Should they also have the same base type?
function Base.:(==)(r1::AbstractNamedUnitRange, r2::AbstractNamedUnitRange)
    return name(r1) == name(r2) && denamed(r1) == denamed(r2)
end
function Base.hash(r::AbstractNamedUnitRange, h::UInt)
    h = hash(Symbol(unspecify_type_parameters(typeof(r))), h)
    # TODO: Double check how this is handling blocking/sector information.
    h = hash(denamed(r), h)
    return hash(name(r), h)
end

# Unit range functionality.
Base.first(r::AbstractNamedUnitRange) = named(first(denamed(r)), name(r))
Base.last(r::AbstractNamedUnitRange) = named(last(denamed(r)), name(r))
Base.length(r::AbstractNamedUnitRange) = named(length(denamed(r)), name(r))
Base.size(r::AbstractNamedUnitRange) = (named(length(denamed(r)), name(r)),)
Base.axes(r::AbstractNamedUnitRange) = (named(only(axes(denamed(r))), name(r)),)
Base.step(r::AbstractNamedUnitRange) = named(step(denamed(r)), name(r))
Base.getindex(r::AbstractNamedUnitRange, I::Int) = getindex_named(r, I)
# Fix ambiguity error.
function Base.getindex(r::AbstractNamedUnitRange, I::AbstractUnitRange{<:Integer})
    return getindex_named(r, I)
end
# Fix ambiguity error.
function Base.getindex(r::AbstractNamedUnitRange, I::Colon)
    return getindex_named(r, I)
end
function Base.getindex(r::AbstractNamedUnitRange, I)
    return getindex_named(r, I)
end
# Fixes `r[begin]`/`r[end]`, since `firstindex` and `lastindex`
# returned named indices.
function Base.getindex(r::AbstractNamedUnitRange, I::AbstractNamedInteger)
    @assert name(r) == name(I)
    return getindex_named(r, denamed(I))
end
Base.isempty(r::AbstractNamedUnitRange) = isempty(denamed(r))

function Base.AbstractUnitRange{Int}(r::AbstractNamedUnitRange)
    return AbstractUnitRange{Int}(denamed(r))
end

Base.oneto(length::AbstractNamedInteger) = named(Base.OneTo(denamed(length)), name(length))
namedoneto(length::Integer, name) = Base.oneto(named(length, name))
Base.iterate(r::AbstractNamedUnitRange) = isempty(r) ? nothing : (first(r), first(r))
function Base.iterate(r::AbstractNamedUnitRange, i)
    i == last(r) && return nothing
    next = named(denamed(i) + denamed(step(r)), name(r))
    return (next, next)
end

function randname(rng::AbstractRNG, r::AbstractNamedUnitRange)
    return named(denamed(r), randname(rng, name(r)))
end

function Base.show(io::IO, r::AbstractNamedUnitRange)
    print(io, "named(", denamed(r), ", ", repr(name(r)), ")")
    return nothing
end

struct NamedColon{Name} <: Function
    name::Name
end
denamed(c::NamedColon) = Colon()
name(c::NamedColon) = c.name
named(::Colon, name) = NamedColon(name)

struct FirstIndex{Arr <: AbstractArray, Dim}
    array::Arr
    dim::Dim
end
Base.to_index(i::FirstIndex) = Int(first(axes(i.array, i.dim)))

struct LastIndex{Arr <: AbstractArray, Dim}
    array::Arr
    dim::Dim
end
Base.to_index(i::LastIndex) = Int(last(axes(i.array, i.dim)))

function Base.getindex(r::AbstractNamedUnitRange, I::FirstIndex)
    return first(r)
end
function Base.getindex(r::AbstractNamedUnitRange, I::LastIndex)
    return last(r)
end
