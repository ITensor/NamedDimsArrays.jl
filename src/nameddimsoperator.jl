using OrderedCollections: OrderedDict
using Random: Random

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
    domain_to_codomain::OrderedDict{Domain, Codomain}
    codomain_to_domain::OrderedDict{Codomain, Domain}
end
function Bijection(domain, codomain)
    pairs = domain .=> codomain
    domain_to_codomain = OrderedDict(pairs)
    codomain_to_domain = OrderedDict(reverse(kv) for kv in pairs)
    return Bijection(domain_to_codomain, codomain_to_domain)
end
function Base.get(b::Bijection, k, default)
    return get(b.domain_to_codomain, k, default)
end
function inverse(b::Bijection)
    return Bijection(b.codomain_to_domain, b.domain_to_codomain)
end
# Both accessors iterate `codomain_to_domain` so that successive calls return
# values in lock-step positional order (codomain[i] paired with domain[i]).
function codomain(b::Bijection)
    return keys(b.codomain_to_domain)
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

# Operator entries for the gram factorizations defined in `tensoralgebra.jl`.
# Placed here because `AbstractNamedDimsOperator` is defined below
# `tensoralgebra.jl` in the include order.
#
# Per-method docstrings are factored out into `const` strings and attached
# inside the `@eval` loop via `@doc`. This keeps the loop body uniform when
# methods need distinct user-facing docs (including jldoctest examples) that
# don't share enough structure to warrant `$($f)`-interpolation.

const _gram_eigh_full_operator_docstring = """
    TensorAlgebra.gram_eigh_full(a::AbstractNamedDimsOperator; kwargs...) -> x

Gram factorization of a Hermitian positive semi-definite named operator
`a`, returning `x` such that `x * x_cod ≈ state(a)`, where `x_cod` is
`conj(x)` with its domain dimension names replaced by the corresponding
codomain names of `a`. `x` carries `a`'s domain dimension names and a
fresh trailing rank name. The codomain and domain partition is taken from
`codomainnames(a)` and `domainnames(a)`.

`kwargs` are forwarded to `TensorAlgebra.gram_eigh_full` on the
underlying named array (e.g. `atol`, `rtol`).

# Examples

```jldoctest
julia> using NamedDimsArrays: namedoneto, operator, replacedimnames, state

julia> using TensorAlgebra: gram_eigh_full

julia> i, j, k, l, aux = namedoneto.((2, 2, 2, 2, 8), ("i", "j", "k", "l", "aux"));

julia> b = randn(aux, i, k);

julia> a = operator(conj(b) * replacedimnames(b, "i" => "j", "k" => "l"), ("i", "k"), ("j", "l"));

julia> x = gram_eigh_full(a);

julia> replacedimnames(x, "j" => "i", "l" => "k") * conj(x) ≈ state(a)
true
```
"""

const _gram_eigh_full_with_pinv_operator_docstring = """
    TensorAlgebra.gram_eigh_full_with_pinv(a::AbstractNamedDimsOperator; kwargs...) -> x, y

Like `TensorAlgebra.gram_eigh_full`, but additionally returns a
named array `y` that is a left inverse of `x`: `y * x ≈ I` on the
rank subspace (equal to the identity when `a` is full rank). The
codomain and domain partition is taken from `codomainnames(a)` and
`domainnames(a)`.

# Examples

```jldoctest
julia> using LinearAlgebra: I

julia> using NamedDimsArrays: dename, dimnames, namedoneto, operator, replacedimnames

julia> using TensorAlgebra: gram_eigh_full_with_pinv

julia> i, j, k, l, aux = namedoneto.((2, 2, 2, 2, 8), ("i", "j", "k", "l", "aux"));

julia> b = randn(aux, i, k);

julia> a = operator(conj(b) * replacedimnames(b, "i" => "j", "k" => "l"), ("i", "k"), ("j", "l"));

julia> x, y = gram_eigh_full_with_pinv(a);

julia> rname = only(setdiff(dimnames(x), ("j", "l")));

julia> reshape(dename(y, (rname, "j", "l")), :, 4) *
       reshape(dename(x, ("j", "l", rname)), 4, :) ≈ I
true
```
"""

for f in (:gram_eigh_full, :gram_eigh_full_with_pinv)
    doc_sym = Symbol("_", f, "_operator_docstring")
    @eval begin
        @doc $doc_sym function TA.$f(a::AbstractNamedDimsOperator; kwargs...)
            return TA.$f(state(a), codomainnames(a), domainnames(a); kwargs...)
        end
    end
end

"""
    Base.one(op::AbstractNamedDimsOperator) -> Id

Return the identity operator with the same codomain/domain names and shape as
`op`. `op` is treated as a shape prototype and is not mutated.

# Examples

```jldoctest
julia> using LinearAlgebra: I

julia> using NamedDimsArrays: dename, namedoneto, operator, state

julia> using TensorAlgebra: matricize

julia> i, j, k, l = namedoneto.((2, 3, 2, 3), ("i", "j", "k", "l"));

julia> op = operator(randn(i, j, k, l), ("i", "j"), ("k", "l"));

julia> Id = one(op);

julia> dename(matricize(state(Id), (i, j) => "row", (k, l) => "col"), ("row", "col")) ≈ I
true
```
"""
function Base.one(op::AbstractNamedDimsOperator)
    co, dom = codomainnames(op), domainnames(op)
    return operator(one(state(op), co, dom), co, dom)
end

# === similar_operator ===
#
# Allocate an operator with the user-supplied side as the domain (input) and
# the codomain (output) derived by `conj`-ing the domain axes and either
# randomizing the codomain names or accepting them explicitly. The 5-arg form
# is canonical; the others fill in defaults.

"""
    similar_operator(prototype, [T,] unnamed_domain_axes, [codomain_names,] domain_names) -> op
    similar_operator(prototype, [T,] named_domain_axes) -> op

Allocate an operator-shaped named array with undefined data, with the
user-supplied side as the domain (input) and the codomain (output) derived by
`conj`-ing the domain axes. Element type defaults to `eltype(prototype)`;
codomain names default to fresh `randname`-generated names. The first form
takes unnamed (raw) axes and explicit names; the second takes already-named
axes and reuses their names as the domain.

The codomain axes are taken to be `conj.(unnamed_domain_axes)` — for plain
axes this is a no-op, while graded axes flip their sector arrows.
"""
function similar_operator(
        prototype, ::Type{T}, unnamed_domain_axes, codomain_names, domain_names
    ) where {T}
    unnamed_codomain_axes = conj.(unnamed_domain_axes)
    codomain_axes = named.(unnamed_codomain_axes, codomain_names)
    domain_axes = named.(unnamed_domain_axes, domain_names)
    raw = similar(prototype, T, (codomain_axes..., domain_axes...))
    return operator(raw, codomain_names, domain_names)
end
function similar_operator(
        prototype, ::Type{T}, unnamed_domain_axes, domain_names
    ) where {T}
    codomain_names = randname.(domain_names)
    return similar_operator(
        prototype, T, unnamed_domain_axes, codomain_names, domain_names
    )
end
function similar_operator(prototype, ::Type{T}, named_domain_axes) where {T}
    return similar_operator(
        prototype, T, denamed.(named_domain_axes), name.(named_domain_axes)
    )
end
function similar_operator(prototype, unnamed_domain_axes, codomain_names, domain_names)
    return similar_operator(
        prototype, eltype(prototype), unnamed_domain_axes, codomain_names, domain_names
    )
end
function similar_operator(prototype, unnamed_domain_axes, domain_names)
    return similar_operator(prototype, eltype(prototype), unnamed_domain_axes, domain_names)
end
function similar_operator(prototype, named_domain_axes)
    return similar_operator(prototype, eltype(prototype), named_domain_axes)
end

# === Random fills for operators ===
#
# Peel down to the concrete storage so `Random.randn!` / `Random.rand!` see the
# runtime eltype. This works around the ITensor `eltype(typeof(::ITensor)) === Any`
# issue, where dispatching on `Type{Any}` would otherwise fail.

function Random.randn!(rng::Random.AbstractRNG, op::AbstractNamedDimsOperator)
    Random.randn!(rng, denamed(state(op)))
    return op
end

function Random.rand!(rng::Random.AbstractRNG, op::AbstractNamedDimsOperator)
    Random.rand!(rng, denamed(state(op)))
    return op
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
