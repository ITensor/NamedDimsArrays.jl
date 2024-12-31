using TypeParameterAccessors: unspecify_type_parameters

abstract type AbstractNamedUnitRange{T,Value<:AbstractUnitRange,Name} <:
              AbstractUnitRange{T} end

# Minimal interface.
dename(r::AbstractNamedUnitRange) = throw(MethodError(dename, Tuple{typeof(r)}))
name(r::AbstractNamedUnitRange) = throw(MethodError(name, Tuple{typeof(r)}))

# This can be customized to output different named integer types,
# such as `namedunitrange(r::AbstractUnitRange, name::IndexName) = Index(r, name)`.
namedunitrange(r::AbstractUnitRange, name) = NamedUnitRange(r, name)

# Shorthand.
named(r::AbstractUnitRange, name) = namedunitrange(r, name)

# Derived interface.
# TODO: Use `Accessors.@set`?
setname(r::AbstractNamedUnitRange, name) = namedunitrange(dename(r), name)

# TODO: Use `TypeParameterAccessors`.
denametype(::Type{<:AbstractNamedUnitRange{<:Any,Value}}) where {Value} = Value
nametype(::Type{<:AbstractNamedUnitRange{<:Any,<:Any,Name}}) where {Name} = Name

# Traits.
isnamed(::Type{<:AbstractNamedUnitRange}) = true

# TODO: Should they also have the same base type?
function Base.:(==)(r1::AbstractNamedUnitRange, r2::AbstractNamedUnitRange)
  return name(r1) == name(r2) && dename(r1) == dename(r2)
end
function Base.hash(r::AbstractNamedUnitRange, h::UInt)
  h = hash(Symbol(unspecify_type_parameters(typeof(r))), h)
  # TODO: Double check how this is handling blocking/sector information.
  h = hash(dename(r), h)
  return hash(name(r), h)
end

# Unit range funcionality.
Base.first(r::AbstractNamedUnitRange) = named(first(dename(r)), name(r))
Base.last(r::AbstractNamedUnitRange) = named(last(dename(r)), name(r))
Base.length(r::AbstractNamedUnitRange) = named(length(dename(r)), name(r))
Base.size(r::AbstractNamedUnitRange) = (named(length(dename(r)), name(r)),)
Base.axes(r::AbstractNamedUnitRange) = (named(only(axes(dename(r))), name(r)),)
Base.step(r::AbstractNamedUnitRange) = named(step(dename(r)), name(r))
Base.getindex(r::AbstractNamedUnitRange, I::Int) = named_getindex(r, I)
# Fix ambiguity error.
function Base.getindex(r::AbstractNamedUnitRange, I::AbstractUnitRange{<:Integer})
  return named_getindex(r, I)
end
# Fix ambiguity error.
function Base.getindex(r::AbstractNamedUnitRange, I::Colon)
  return named_getindex(r, I)
end
function Base.getindex(r::AbstractNamedUnitRange, I)
  return named_getindex(r, I)
end
Base.isempty(r::AbstractNamedUnitRange) = isempty(dename(r))

function Base.AbstractUnitRange{Int}(r::AbstractNamedUnitRange)
  return AbstractUnitRange{Int}(dename(r))
end

Base.oneto(length::AbstractNamedInteger) = named(Base.OneTo(dename(length)), name(length))
namedoneto(length::Integer, name) = Base.oneto(named(length, name))
Base.iterate(r::AbstractNamedUnitRange) = isempty(r) ? nothing : (first(r), first(r))
function Base.iterate(r::AbstractNamedUnitRange, i)
  i == last(r) && return nothing
  next = named(dename(i) + dename(step(r)), name(r))
  return (next, next)
end

function randname(rng::AbstractRNG, r::AbstractNamedUnitRange)
  return named(dename(r), randname(name(r)))
end

function Base.show(io::IO, r::AbstractNamedUnitRange)
  print(io, "named(", dename(r), ", ", repr(name(r)), ")")
  return nothing
end

struct NamedColon{Name} <: Function
  name::Name
end
dename(c::NamedColon) = Colon()
name(c::NamedColon) = c.name
named(::Colon, name) = NamedColon(name)
