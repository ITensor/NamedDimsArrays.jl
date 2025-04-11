using TypeParameterAccessors: unspecify_type_parameters

abstract type AbstractNamedInteger{Value,Name} <: Integer end

# Minimal interface.
dename(i::AbstractNamedInteger) = throw(MethodError(dename, Tuple{typeof(i)}))
name(i::AbstractNamedInteger) = throw(MethodError(name, Tuple{typeof(i)}))

# This can be customized to output different named integer types,
# such as `namedinteger(i::Integer, name::IndexName) = IndexInteger(i, name)`.
namedinteger(i::Integer, name) = NamedInteger(i, name)

# Shorthand.
named(i::Integer, name) = namedinteger(i, name)

# Derived interface.
# TODO: Use `Accessors.@set`?
setname(i::AbstractNamedInteger, name) = named(dename(i), name)
# TODO: Use `Accessors.@set`?
setvalue(i::AbstractNamedInteger, value) = named(value, name(i))

# TODO: Use `TypeParameterAccessors`.
denametype(::Type{<:AbstractNamedInteger{Value}}) where {Value} = Value
nametype(::Type{<:AbstractNamedInteger{<:Any,Name}}) where {Name} = Name

# Traits
isnamed(::Type{<:AbstractNamedInteger}) = true

# TODO: Should they also have the same base type?
function Base.:(==)(i1::AbstractNamedInteger, i2::AbstractNamedInteger)
  return name(i1) == name(i2) && dename(i1) == dename(i2)
end
function Base.hash(i::AbstractNamedInteger, h::UInt)
  h = hash(Symbol(unspecify_type_parameters(typeof(i))), h)
  h = hash(dename(i), h)
  return hash(name(i), h)
end

abstract type AbstractName end
# TODO: Decide if this is a good definition, probably not.
# name(n::AbstractName) = throw(MethodError(name, Tuple{typeof(n)}))
Base.getindex(n::AbstractName, I) = named(I, name(n))

struct Name{Value} <: AbstractName
  value::Value
end
name(n::Name) = n.value

# vcat that works with combinations of tuples
# and vectors.
generic_vcat(v1, v2) = vcat(v1, v2)
generic_vcat(v1::Tuple, v2) = vcat([v1...], v2)
generic_vcat(v1, v2::Tuple) = vcat(v1, [v2...])
generic_vcat(v1::Tuple, v2::Tuple) = (v1..., v2...)

struct FusedNames{Names} <: AbstractName
  names::Names
end
fusednames(name1, name2) = FusedNames((name1, name2))
function fusednames(name1::FusedNames, name2::FusedNames)
  return FusedNames(generic_vcat(name1.names, name2.names))
end
fusednames(name1, name2::FusedNames) = fusednames(FusedNames((name1,)), name2)
fusednames(name1::FusedNames, name2) = fusednames(name1, FusedNames((name2,)))

function Base.:(==)(n1::FusedNames, n2::FusedNames)
  return mapreduce(==,&,n1.names,n2.names)
end

# Integer interface
# TODO: Should this make a random name, or require defining a way
# to combine names?
function Base.:*(i1::AbstractNamedInteger, i2::AbstractNamedInteger)
  return named(dename(i1) * dename(i2), fusednames(name(i1), name(i2)))
end
Base.:-(i::AbstractNamedInteger) = setvalue(i, -dename(i))

## TODO: Support this, we need to define `NamedFloat`, `NamedReal`, `NamedNumber`, etc.
## This is used in `LinearAlgebra.norm`, for now we just overload that directly.
## Here, named numbers are treated as unitful, so multiplying them
## with unnamed numbers means the result inherits the name.
## function Base.:*(i1::AbstractNamedInteger, i2::Number)
##   return named(dename(i1) * i2, name(i1))
## end

# For the sake of generic code, the name is ignored.
# Used in `AbstractArray` `Base.show`.
# TODO: See if we can delete this.
Base.:+(i1::Int, i2::AbstractNamedInteger) = i1 + dename(i2)

Base.:*(i1::Int, i2::AbstractNamedInteger) = named(i1 * dename(i2), name(i2))

Base.zero(i::AbstractNamedInteger) = setvalue(i, zero(dename(i)))
Base.one(i::AbstractNamedInteger) = setvalue(i, one(dename(i)))
Base.signbit(i::AbstractNamedInteger) = signbit(dename(i))
Base.unsigned(i::AbstractNamedInteger) = setvalue(i, unsigned(dename(i)))
function Base.string(i::AbstractNamedInteger; kwargs...)
  return "named($(string(dename(i); kwargs...)), $(repr(name(i))))"
end

Base.Int(i::AbstractNamedInteger) = Int(dename(i))

struct NameMismatch <: Exception
  message::String
end
NameMismatch() = NameMismatch("")

function randname(rng::AbstractRNG, i::AbstractNamedInteger)
  return named(dename(i), randname(name(i)))
end

# Used in bounds checking when indexing with named dimensions.
function Base.:<(i1::AbstractNamedInteger, i2::AbstractNamedInteger)
  name(i1) == name(i2) || throw(NameMismatch("Mismatched names $(name(i1)), $(name(i2))"))
  return dename(i1) < dename(i2)
end

function Base.show(io::IO, r::AbstractNamedInteger)
  print(io, "named(", dename(r), ", ", repr(name(r)), ")")
  return nothing
end
