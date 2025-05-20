# Named dimension operator minimal interface.

# Choi state representation of the named operator.
# https://en.wikipedia.org/wiki/Choi%E2%80%93Jamio%C5%82kowski_isomorphism
state(a) = throw(MethodError(state, (a,)))
# Operator representation of the named state given pairs of named codomain and domain indices.
function operator(a, codomain_domain_pairs)
  throw(MethodError(operator, (a, codomain_domain_pairs)))
end

# Get the named domain indices of the operator.
domain(a) = throw(MethodError(domain, (a,)))
# Get the named codomain indices of the operator.
codomain(a) = throw(MethodError(codomain, (a,)))

# Given a named codomain index, return the corresponding named domain index.
# If it doesn't exist, return the index itself.
get_domain_ind(a, i) = throw(MethodError(get_domain_ind, (a, i)))
# Given a named domain index, return the corresponding named codomain index.
# If it doesn't exist, return the index itself.
get_codomain_ind(a, i) = throw(MethodError(get_codomain_ind, (a, i)))

# TODO: Should this be `adjoint`?
function dag(a::AbstractNamedDimsArray, inds_map)
  a = conj(a)
  a′ = mapnameddimsindices(a) do i
    return get(inds_map, i, i)
  end
  return a′
end

function apply(x::AbstractNamedDimsArray, y::AbstractNamedDimsArray)
  xy = x * y
  return mapnameddimsindices(xy) do i
    return get_domain_ind(x, i)
  end
end

function apply_dag(x::AbstractNamedDimsArray, y::AbstractNamedDimsArray)
  xy = x * y
  return mapnameddimsindices(xy) do i
    return get_codomain_ind(y, i)
  end
end

function Base.transpose(a::AbstractNamedDimsArray)
  c = codomain(a)
  d = domain(a)
  a_map = merge(Dict(c .=> d), Dict(d .=> c))
  a′ = mapnameddimsindices(state(a)) do i
    return get(a_map, i, i)
  end
  return operator(a′, c .=> d)
end

function Base.adjoint(a::AbstractNamedDimsArray)
  return transpose(conj(a))
end

function product(x::AbstractNamedDimsArray, y::AbstractNamedDimsArray)
  c = codomain(x)
  d = domain(x)
  c′ = randname.(c)
  x′_map = merge(Dict(c .=> c′), Dict(d .=> c))
  x′ = mapnameddimsindices(parent(x)) do i
    return get(x′_map, i, i)
  end
  x′y = x′ * parent(y)
  x′y_map = Dict(c′ .=> c)
  xy = mapnameddimsindices(x′y) do i
    return get(x′y_map, i, i)
  end
  return operator(xy, c .=> d)
end

struct Bijection{Codomain,Domain} <: AbstractDict{Domain,Codomain}
  domain_to_codomain::Dict{Domain,Codomain}
  codomain_to_domain::Dict{Codomain,Domain}
end
function Bijection(domain_codomain_pairs)
  domain_to_codomain = Dict(domain_codomain_pairs)
  codomain_to_domain = Dict(reverse(kv) for kv in domain_codomain_pairs)
  return Bijection(domain_to_codomain, codomain_to_domain)
end
function Base.get(b::Bijection, k, default)
  return get(b.domain_to_codomain, k, default)
end
function inverse(b::Bijection)
  return Bijection(b.codomain_to_domain, b.domain_to_codomain)
end
function domain(b::Bijection)
  return keys(b.domain_to_codomain)
end
function codomain(b::Bijection)
  return values(b.domain_to_codomain)
end
Base.iterate(b::Bijection) = iterate(b.domain_to_codomain)
Base.iterate(b::Bijection, state) = iterate(b.domain_to_codomain, state)
Base.length(b::Bijection) = length(b.domain_to_codomain)

# Bijection between the named codomain and domain indices of the operator.
# It should act like a dictionary from the domain to the codomain,
# but then under `inverse` it should act like a dictionary from the codomain to the domain.
# Primarily it should define `get`.
function inds_map(a)
  return Bijection(domain(a) .=> codomain(a))
end

abstract type AbstractNamedDimsOperator{T,N} <: AbstractNamedDimsArray{T,N} end

struct NamedDimsOperator{T,N,P<:AbstractNamedDimsArray{T,N},D,C} <:
       AbstractNamedDimsOperator{T,N}
  parent::P
  domain_codomain_bijection::Bijection{D,C}
end

inds_map(a::NamedDimsOperator) = getfield(a, :domain_codomain_bijection)

function NamedDimsOperator(a::AbstractNamedDimsArray, domain_codomain_pairs)
  domain = to_nameddimsindices(a, first.(domain_codomain_pairs))
  codomain = to_nameddimsindices(a, last.(domain_codomain_pairs))
  return NamedDimsOperator(a, Bijection(domain .=> codomain))
end

Base.parent(a::NamedDimsOperator) = getfield(a, :parent)

using TypeParameterAccessors: TypeParameterAccessors
function TypeParameterAccessors.parenttype(type::Type{<:NamedDimsOperator})
  fieldtype(type, :parent)
end

NamedDimsArrays.nameddimsindices(a::NamedDimsOperator) = nameddimsindices(parent(a))
NamedDimsArrays.dename(a::NamedDimsOperator) = dename(parent(a))

function NamedDimsArrays.constructorof_nameddimsarray(type::Type{<:NamedDimsOperator})
  return constructorof_nameddimsarray(parenttype(type))
end

state(a::NamedDimsOperator) = parent(a)
function operator(a::NamedDimsArray, domain_codomain_pairs)
  NamedDimsOperator(a, domain_codomain_pairs)
end

function operator(a::AbstractArray, codomain, domain)
  na = nameddimsarray(a, (codomain..., domain...))
  return operator(na, domain .=> codomain)
end

# TODO: Make abstract?
domain(a::NamedDimsOperator) = domain(inds_map(a))
# TODO: Make abstract?
codomain(a::NamedDimsOperator) = codomain(inds_map(a))

# TODO: Make abstract?
function get_domain_ind(a::NamedDimsOperator, i)
  return get(inverse(inds_map(a)), i, i)
end
# TODO: Make abstract?
function get_codomain_ind(a::NamedDimsOperator, i)
  return get(inds_map(a), i, i)
end
