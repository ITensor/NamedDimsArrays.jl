using LinearAlgebra: LinearAlgebra
using TensorAlgebra:
    TensorAlgebra,
    blockedperm,
    contract,
    contract!,
    contractadd!,
    eigen,
    eigvals,
    factorize,
    left_null,
    left_orth,
    left_polar,
    lq,
    matricize,
    orth,
    permmortar,
    polar,
    qr,
    right_null,
    right_orth,
    right_polar,
    svd,
    svdvals,
    unmatricize
using TensorAlgebra.BaseExtensions: BaseExtensions
using TupleTools: TupleTools

function TensorAlgebra.contractadd!(
        a_dest::AbstractNamedDimsArray,
        a1::AbstractNamedDimsArray,
        a2::AbstractNamedDimsArray,
        α::Number,
        β::Number,
    )
    contractadd!(
        dename(a_dest),
        nameddimsindices(a_dest),
        dename(a1),
        nameddimsindices(a1),
        dename(a2),
        nameddimsindices(a2),
        α,
        β,
    )
    return a_dest
end

function TensorAlgebra.contract!(
        a_dest::AbstractNamedDimsArray, a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray
    )
    return contractadd!(a_dest, a1, a2, true, false)
end

function TensorAlgebra.contract(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
    a_dest, nameddimsindices_dest = contract(
        dename(a1), nameddimsindices(a1), dename(a2), nameddimsindices(a2)
    )
    nameddimstype = combine_nameddimsarraytype(
        constructorof(typeof(a1)), constructorof(typeof(a2))
    )
    return nameddimstype(a_dest, nameddimsindices_dest)
end

function Base.:*(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
    return contract(a1, a2)
end

# Left associative fold/reduction.
# Circumvent Base definitions:
# ```julia
# *(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
# *(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
# ```
# that optimize matrix multiplication sequence.
function Base.:*(
        a1::AbstractNamedDimsArray,
        a2::AbstractNamedDimsArray,
        a3::AbstractNamedDimsArray,
        a_rest::AbstractNamedDimsArray...,
    )
    return *(*(a1, a2), a3, a_rest...)
end

function LinearAlgebra.mul!(
        a_dest::AbstractNamedDimsArray,
        a1::AbstractNamedDimsArray,
        a2::AbstractNamedDimsArray,
        α::Number,
        β::Number,
    )
    contractadd!(a_dest, a1, a2, α, β)
    return a_dest
end

function LinearAlgebra.mul!(
        a_dest::AbstractNamedDimsArray, a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray
    )
    contract!(a_dest, a1, a2)
    return a_dest
end

function TensorAlgebra.blockedperm(na::AbstractNamedDimsArray, nameddim_blocks::Tuple...)
    dimname_blocks = map(group -> to_nameddimsindices(na, group), nameddim_blocks)
    nameddimsindices_a = nameddimsindices(na)
    perms = map(dimname_blocks) do dimname_block
        return BaseExtensions.indexin(dimname_block, nameddimsindices_a)
    end
    return permmortar(perms)
end

# i, j, k, l = named.((2, 2, 2, 2), ("i", "j", "k", "l"))
# a = randn(i, j, k, l)
# matricize(a, (i, k) => "a")
# matricize(a, (i, k) => "a", (j, l) => "b")
# TODO: Rewrite in terms of `matricize(a, .., (1, 3))` interface.
function TensorAlgebra.matricize(na::AbstractNamedDimsArray, fusions::Vararg{Pair, 2})
    nameddimsindices_fuse = map(group -> to_nameddimsindices(na, group), first.(fusions))
    nameddimsindices_fused = last.(fusions)
    if sum(length, nameddimsindices_fuse) < ndims(na)
        # Not all names are specified
        nameddimsindices_unspecified = setdiff(nameddimsindices(na), nameddimsindices_fuse...)
        nameddimsindices_fuse = vcat(
            tuple.(nameddimsindices_unspecified), collect(nameddimsindices_fuse)
        )
        nameddimsindices_fused = vcat(
            nameddimsindices_unspecified, collect(nameddimsindices_fused)
        )
    end
    perm = blockedperm(na, nameddimsindices_fuse...)
    a_fused = matricize(dename(na), perm)
    return nameddimsarray(a_fused, nameddimsindices_fused)
end

function TensorAlgebra.unmatricize(na::AbstractNamedDimsArray, splitters::Vararg{Pair, 2})
    splitters = to_nameddimsindices(na, first.(splitters)) .=> last.(splitters)
    split_namedlengths = last.(splitters)
    splitters_unnamed = map(splitters) do splitter
        fused_name, split_namedlengths = splitter
        fused_dim = findfirst(isequal(fused_name), nameddimsindices(na))
        split_lengths = unname.(split_namedlengths)
        return fused_dim => split_lengths
    end
    blocked_axes = last.(TupleTools.sort(splitters_unnamed; by = first))
    a_split = unmatricize(dename(na), blocked_axes...)
    names_split = Any[tuple.(nameddimsindices(na))...]
    for splitter in splitters
        fused_name, split_namedlengths = splitter
        fused_dim = findfirst(isequal(fused_name), nameddimsindices(na))
        split_names = name.(split_namedlengths)
        names_split[fused_dim] = split_names
    end
    names_split = reduce((x, y) -> (x..., y...), names_split)
    return nameddimsarray(a_split, names_split)
end

for f in [
        :factorize, :left_orth, :left_polar, :lq, :orth, :polar, :qr, :right_orth, :right_polar,
    ]
    @eval begin
        function TensorAlgebra.$f(
                a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
            )
            codomain = to_nameddimsindices(a, dimnames_codomain)
            domain = to_nameddimsindices(a, dimnames_domain)
            x_unnamed, y_unnamed = $f(dename(a), nameddimsindices(a), codomain, domain; kwargs...)
            name_x = randname(dimnames(a, 1))
            name_y = name_x
            namedindices_x = named(last(axes(x_unnamed)), name_x)
            namedindices_y = named(first(axes(y_unnamed)), name_y)
            nameddimsindices_x = (codomain..., namedindices_x)
            nameddimsindices_y = (namedindices_y, domain...)
            x = nameddimsarray(x_unnamed, nameddimsindices_x)
            y = nameddimsarray(y_unnamed, nameddimsindices_y)
            return x, y
        end
        function TensorAlgebra.$f(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
            codomain = to_nameddimsindices(a, dimnames_codomain)
            domain = setdiff(nameddimsindices(a), codomain)
            return $f(a, codomain, domain; kwargs...)
        end
    end
end

# Overload LinearAlgebra functions where relevant.
function LinearAlgebra.qr(a::AbstractNamedDimsArray, args...; kwargs...)
    return TensorAlgebra.qr(a, args...; kwargs...)
end
function LinearAlgebra.lq(a::AbstractNamedDimsArray, args...; kwargs...)
    return TensorAlgebra.lq(a, args...; kwargs...)
end
function LinearAlgebra.factorize(a::AbstractNamedDimsArray, args...; kwargs...)
    return TensorAlgebra.factorize(a, args...; kwargs...)
end

#
# Non-binary factorizations.
#

function TensorAlgebra.svd(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_nameddimsindices(a, dimnames_codomain)
    domain = to_nameddimsindices(a, dimnames_domain)
    u_unnamed, s_unnamed, v_unnamed = svd(
        dename(a), nameddimsindices(a), codomain, domain; kwargs...
    )
    name_u = randname(dimnames(a, 1))
    name_v = randname(dimnames(a, 1))
    namedindices_u = named(last(axes(u_unnamed)), name_u)
    namedindices_v = named(first(axes(v_unnamed)), name_v)
    nameddimsindices_u = (codomain..., namedindices_u)
    nameddimsindices_s = (namedindices_u, namedindices_v)
    nameddimsindices_v = (namedindices_v, domain...)
    u = nameddimsarray(u_unnamed, nameddimsindices_u)
    s = nameddimsarray(s_unnamed, nameddimsindices_s)
    v = nameddimsarray(v_unnamed, nameddimsindices_v)
    return u, s, v
end
function TensorAlgebra.svd(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return svd(
        a,
        dimnames_codomain,
        setdiff(nameddimsindices(a), to_nameddimsindices(a, dimnames_codomain));
        kwargs...,
    )
end
function LinearAlgebra.svd(a::AbstractNamedDimsArray, args...; kwargs...)
    return TensorAlgebra.svd(a, args...; kwargs...)
end

function TensorAlgebra.svdvals(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    return svdvals(
        dename(a),
        nameddimsindices(a),
        to_nameddimsindices(a, dimnames_codomain),
        to_nameddimsindices(a, dimnames_domain);
        kwargs...,
    )
end
function TensorAlgebra.svdvals(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    codomain = to_nameddimsindices(a, dimnames_codomain)
    domain = setdiff(nameddimsindices(a), codomain)
    return svdvals(a, codomain, domain; kwargs...)
end
function LinearAlgebra.svdvals(a::AbstractNamedDimsArray, args...; kwargs...)
    return TensorAlgebra.svdvals(a, args...; kwargs...)
end

function TensorAlgebra.eigen(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_nameddimsindices(a, dimnames_codomain)
    domain = to_nameddimsindices(a, dimnames_domain)
    d_unnamed, v_unnamed = eigen(dename(a), nameddimsindices(a), codomain, domain; kwargs...)
    name_d = randname(dimnames(a, 1))
    name_d′ = randname(name_d)
    name_v = name_d
    namedindices_d = named(last(axes(d_unnamed)), name_d)
    namedindices_d′ = named(first(axes(d_unnamed)), name_d′)
    namedindices_v = named(last(axes(v_unnamed)), name_v)
    nameddimsindices_d = (namedindices_d′, namedindices_d)
    nameddimsindices_v = (domain..., namedindices_v)
    d = nameddimsarray(d_unnamed, nameddimsindices_d)
    v = nameddimsarray(v_unnamed, nameddimsindices_v)
    return d, v
end
function LinearAlgebra.eigen(a::AbstractNamedDimsArray, args...; kwargs...)
    return TensorAlgebra.eigen(a, args...; kwargs...)
end

function TensorAlgebra.eigvals(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_nameddimsindices(a, dimnames_codomain)
    domain = to_nameddimsindices(a, dimnames_domain)
    return eigvals(dename(a), nameddimsindices(a), codomain, domain; kwargs...)
end
function LinearAlgebra.eigvals(a::AbstractNamedDimsArray, args...; kwargs...)
    return TensorAlgebra.eigvals(a, args...; kwargs...)
end

function TensorAlgebra.left_null(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_nameddimsindices(a, dimnames_codomain)
    domain = to_nameddimsindices(a, dimnames_domain)
    n_unnamed = left_null(dename(a), nameddimsindices(a), codomain, domain; kwargs...)
    name_n = randname(dimnames(a, 1))
    namedindices_n = named(last(axes(n_unnamed)), name_n)
    nameddimsindices_n = (codomain..., namedindices_n)
    return nameddimsarray(n_unnamed, nameddimsindices_n)
end
function TensorAlgebra.left_null(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    codomain = to_nameddimsindices(a, dimnames_codomain)
    domain = setdiff(nameddimsindices(a), codomain)
    return left_null(a, codomain, domain; kwargs...)
end

function TensorAlgebra.right_null(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_nameddimsindices(a, dimnames_codomain)
    domain = to_nameddimsindices(a, dimnames_domain)
    n_unnamed = right_null(dename(a), nameddimsindices(a), codomain, domain; kwargs...)
    name_n = randname(dimnames(a, 1))
    namedindices_n = named(first(axes(n_unnamed)), name_n)
    nameddimsindices_n = (namedindices_n, domain...)
    return nameddimsarray(n_unnamed, nameddimsindices_n)
end
function TensorAlgebra.right_null(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    codomain = to_nameddimsindices(a, dimnames_codomain)
    domain = setdiff(nameddimsindices(a), codomain)
    return right_null(a, codomain, domain; kwargs...)
end

const MATRIX_FUNCTIONS = [
    :exp,
    :cis,
    :log,
    :sqrt,
    :cbrt,
    :cos,
    :sin,
    :tan,
    :csc,
    :sec,
    :cot,
    :cosh,
    :sinh,
    :tanh,
    :csch,
    :sech,
    :coth,
    :acos,
    :asin,
    :atan,
    :acsc,
    :asec,
    :acot,
    :acosh,
    :asinh,
    :atanh,
    :acsch,
    :asech,
    :acoth,
]

for f in MATRIX_FUNCTIONS
    @eval begin
        function Base.$f(
                a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
            )
            codomain = to_nameddimsindices(a, dimnames_codomain)
            domain = to_nameddimsindices(a, dimnames_domain)
            fa_unnamed = TensorAlgebra.$f(
                dename(a), nameddimsindices(a), codomain, domain; kwargs...
            )
            return nameddimsarray(fa_unnamed, (codomain..., domain...))
        end
    end
end
