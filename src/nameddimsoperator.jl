# Named dimension operator minimal interface.

# Choi state representation of the named operator.
# https://en.wikipedia.org/wiki/Choi%E2%80%93Jamio%C5%82kowski_isomorphism
state(a) = throw(MethodError(state, (a,)))
# Operator representation of the named state given pairs of named codomain and domain indices.
operator(a, codomain, domain) = throw(MethodError(operator, (a, codomain, domain)))

# Get the codomain dimension names of the operator.
codomainnames(a) = throw(MethodError(codomainnames, (a,)))
# Get the domain dimension names of the operator.
domainnames(a) = throw(MethodError(domainnames, (a,)))

# Given a domain dimension name, return the corresponding codomain dimension name.
# If it doesn't exist, return the index itself.
get_codomain_name(a, i) = throw(MethodError(get_codomain_name, (a, i)))
# Given a codomain dimension name, return the corresponding domain dimension name.
# If it doesn't exist, return the index itself.
get_domain_name(a, i) = throw(MethodError(get_domain_name, (a, i)))

function apply(x::AbstractNamedDimsArray, y::AbstractNamedDimsArray)
    xy = x * y
    return mapdimnames(xy) do i
        return get_domain_name(x, i)
    end
end

function apply_dag(x::AbstractNamedDimsArray, y::AbstractNamedDimsArray)
    xy = x * y
    return mapdimnames(xy) do i
        return get_codomain_name(y, i)
    end
end

# TODO: Define versions that accept codomain and domain names,
# i.e. `transpose(a, codomain, domain)` and `adjoint(a, codomain, domain)` (?).
function Base.transpose(a::AbstractNamedDimsArray)
    c = codomainnames(a)
    d = domainnames(a)
    a_map = merge(Dict(c .=> d), Dict(d .=> c))
    a′ = mapdimnames(state(a)) do i
        return get(a_map, i, i)
    end
    return operator(a′, c, d)
end
function Base.adjoint(a::AbstractNamedDimsArray)
    return transpose(conj(a))
end

function product(x::AbstractNamedDimsArray, y::AbstractNamedDimsArray)
    c = codomainnames(x)
    d = domainnames(x)
    c′ = randname.(c)
    x′_map = merge(Dict(c .=> c′), Dict(d .=> c))
    x′ = mapdimnames(parent(x)) do i
        return get(x′_map, i, i)
    end
    x′y = x′ * parent(y)
    x′y_map = Dict(c′ .=> c)
    xy = mapdimnames(x′y) do i
        return get(x′y_map, i, i)
    end
    return operator(xy, c, d)
end

struct Bijection{Codomain, Domain} <: AbstractDict{Domain, Codomain}
    domain_to_codomain::Dict{Domain, Codomain}
    codomain_to_domain::Dict{Codomain, Domain}
end
function Bijection(domain, codomain)
    pairs = domain .=> codomain
    domain_to_codomain = Dict(pairs)
    codomain_to_domain = Dict(reverse(kv) for kv in pairs)
    return Bijection(domain_to_codomain, codomain_to_domain)
end
function Base.get(b::Bijection, k, default)
    return get(b.domain_to_codomain, k, default)
end
function inverse(b::Bijection)
    return Bijection(b.codomain_to_domain, b.domain_to_codomain)
end
function codomain(b::Bijection)
    return values(b.domain_to_codomain)
end
function domain(b::Bijection)
    return values(b.codomain_to_domain)
end
Base.iterate(b::Bijection) = iterate(b.domain_to_codomain)
Base.iterate(b::Bijection, state) = iterate(b.domain_to_codomain, state)
Base.length(b::Bijection) = length(b.domain_to_codomain)

abstract type AbstractNamedDimsOperator{T, N} <: AbstractNamedDimsArray{T, N} end

state(a::AbstractNamedDimsArray) = a
dimnames(a::AbstractNamedDimsOperator) = dimnames(state(a))

# TODO: Unify these two functions.
function operator(a::AbstractArray, codomain, domain)
    na = nameddims(a, (codomain..., domain...))
    return operator(na, codomain, domain)
end
function operator(a::AbstractNamedDimsArray, codomain, domain)
    return NamedDimsOperator(a, codomain, domain)
end

# This helps preserve the NamedDimsArray type when multiplying,
# for example when a NamedDimsOperator wraps an ITensor.
Base.:*(a::AbstractNamedDimsOperator, b::AbstractNamedDimsOperator) = state(a) * state(b)
Base.:*(a::AbstractNamedDimsOperator, b::AbstractNamedDimsArray) = state(a) * state(b)
Base.:*(a::AbstractNamedDimsArray, b::AbstractNamedDimsOperator) = state(a) * state(b)

for f in MATRIX_FUNCTIONS
    @eval begin
        function Base.$f(a::AbstractNamedDimsOperator)
            c = codomainnames(a)
            d = domainnames(a)
            return operator($f(state(a), c, d), c, d)
        end
    end
end
struct NamedDimsOperator{T, N, P <: AbstractNamedDimsArray{T, N}, D, C} <:
    AbstractNamedDimsOperator{T, N}
    parent::P
    dimnames_bijection::Bijection{D, C}
end

state(a::NamedDimsOperator) = a.parent
Base.parent(a::NamedDimsOperator) = state(a)
denamed(a::NamedDimsOperator) = denamed(state(a))

function NamedDimsOperator(a::AbstractNamedDimsArray, codomainnames, domainnames)
    return NamedDimsOperator(a, Bijection(domainnames, codomainnames))
end

using TypeParameterAccessors: TypeParameterAccessors, parenttype
function TypeParameterAccessors.parenttype(type::Type{<:NamedDimsOperator})
    return fieldtype(type, :parent)
end
statetype(type::Type{<:NamedDimsOperator}) = parenttype(type)

function nameddimsof(a::NamedDimsOperator, b::AbstractArray)
    return NamedDimsOperator(nameddimsof(state(a), b), a.dimnames_bijection)
end
function nameddimsconstructorof(type::Type{<:NamedDimsOperator})
    return nameddimsconstructorof(statetype(type))
end

codomainnames(a::NamedDimsOperator) = codomain(a.dimnames_bijection)
domainnames(a::NamedDimsOperator) = domain(a.dimnames_bijection)

function get_codomain_name(a::NamedDimsOperator, i)
    return get(a.dimnames_bijection, i, i)
end
function get_domain_name(a::NamedDimsOperator, i)
    return get(inverse(a.dimnames_bijection), i, i)
end
