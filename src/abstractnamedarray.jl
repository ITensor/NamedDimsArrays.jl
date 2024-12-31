using TypeParameterAccessors: unspecify_type_parameters

abstract type AbstractNamedArray{T,N,Value<:AbstractArray,Name} <: AbstractArray{T,N} end

const AbstractNamedVector{T,Value<:AbstractVector,Name} = AbstractNamedArray{T,1,Value,Name}
const AbstractNamedMatrix{T,Value<:AbstractVector,Name} = AbstractNamedArray{T,2,Value,Name}

# Minimal interface.
dename(a::AbstractNamedArray) = throw(MethodError(dename, Tuple{typeof(a)}))
name(a::AbstractNamedArray) = throw(MethodError(name, Tuple{typeof(a)}))

# This can be customized to output different named integer types,
# such as `namedarray(a::AbstractArray, name::IndexName) = Index(a, name)`.
namedarray(a::AbstractArray, name) = NamedArray(a, name)

# Shorthand.
named(a::AbstractArray, name) = namedarray(a, name)

# Derived interface.
# TODO: Use `Accessors.@set`?
setname(a::AbstractNamedArray, name) = namedarray(dename(a), name)

# TODO: Use `TypeParameterAccessors`.
denametype(::Type{<:AbstractNamedArray{<:Any,<:Any,Value}}) where {Value} = Value
nametype(::Type{<:AbstractNamedArray{<:Any,<:Any,<:Any,Name}}) where {Name} = Name

# Traits.
isnamed(::Type{<:AbstractNamedArray}) = true

# TODO: Should they also have the same base type?
function Base.:(==)(a1::AbstractNamedArray, a2::AbstractNamedArray)
  return name(a1) == name(a2) && dename(a1) == dename(a2)
end
function Base.hash(a::AbstractNamedArray, h::UInt)
  h = hash(Symbol(unspecify_type_parameters(typeof(a))), h)
  # TODO: Double check how this is handling blocking/sector information.
  h = hash(dename(a), h)
  return hash(name(a), h)
end

named_getindex(a::AbstractArray, I...) = named(getindex(dename(a), I...), name(a))

# Array funcionality.
Base.size(a::AbstractNamedArray) = map(s -> named(s, name(a)), size(dename(a)))
Base.axes(a::AbstractNamedArray) = map(s -> named(s, name(a)), axes(dename(a)))
Base.eachindex(a::AbstractNamedArray) = eachindex(dename(a))
function Base.getindex(a::AbstractNamedArray{<:Any,N}, I::Vararg{Int,N}) where {N}
  return named_getindex(a, I...)
end
function Base.getindex(a::AbstractNamedArray, I::Int)
  return named_getindex(a, I)
end
Base.isempty(a::AbstractNamedArray) = isempty(dename(a))

## function Base.AbstractArray{Int}(a::AbstractNamedArray)
##   return AbstractArray{Int}(dename(a))
## end
## 
## Base.iterate(a::AbstractNamedArray) = isempty(a) ? nothing : (first(a), first(a))
## function Base.iterate(a::AbstractNamedArray, i)
##   i == last(a) && return nothing
##   next = named(dename(i) + dename(step(a)), name(a))
##   return (next, next)
## end

function randname(ang::AbstractRNG, a::AbstractNamedArray)
  return named(dename(a), randname(name(a)))
end

function Base.show(io::IO, a::AbstractNamedArray)
  print(io, "named(", dename(a), ", ", repr(name(a)), ")")
  return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::AbstractNamedArray)
  print(io, "named(\n")
  show(io, mime, dename(a))
  print(io, ",\n ", repr(name(a)), ")")
  return nothing
end
