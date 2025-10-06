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
        inds(a_dest),
        dename(a1),
        inds(a1),
        dename(a2),
        inds(a2),
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
    a_dest, inds_dest = contract(
        dename(a1), inds(a1), dename(a2), inds(a2)
    )
    nameddimstype = combine_nameddimstype(
        constructorof(typeof(a1)), constructorof(typeof(a2))
    )
    return nameddimstype(a_dest, inds_dest)
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
    dimname_blocks = map(group -> to_inds(na, group), nameddim_blocks)
    inds_a = inds(na)
    perms = map(dimname_blocks) do dimname_block
        return BaseExtensions.indexin(dimname_block, inds_a)
    end
    return permmortar(perms)
end

# i, j, k, l = named.((2, 2, 2, 2), ("i", "j", "k", "l"))
# a = randn(i, j, k, l)
# matricize(a, (i, k) => "a")
# matricize(a, (i, k) => "a", (j, l) => "b")
# TODO: Rewrite in terms of `matricize(a, .., (1, 3))` interface.
function TensorAlgebra.matricize(na::AbstractNamedDimsArray, fusions::Vararg{Pair, 2})
    inds_fuse = map(group -> to_inds(na, group), first.(fusions))
    inds_fused = last.(fusions)
    if sum(length, inds_fuse) < ndims(na)
        # Not all names are specified
        inds_unspecified = setdiff(inds(na), inds_fuse...)
        inds_fuse = vcat(
            tuple.(inds_unspecified), collect(inds_fuse)
        )
        inds_fused = vcat(
            inds_unspecified, collect(inds_fused)
        )
    end
    perm = blockedperm(na, inds_fuse...)
    a_fused = matricize(dename(na), perm)
    return nameddims(a_fused, inds_fused)
end

function TensorAlgebra.unmatricize(na::AbstractNamedDimsArray, splitters::Vararg{Pair, 2})
    splitters = to_inds(na, first.(splitters)) .=> last.(splitters)
    split_namedlengths = last.(splitters)
    splitters_unnamed = map(splitters) do splitter
        fused_name, split_namedlengths = splitter
        fused_dim = findfirst(isequal(fused_name), inds(na))
        split_lengths = unname.(split_namedlengths)
        return fused_dim => split_lengths
    end
    blocked_axes = last.(TupleTools.sort(splitters_unnamed; by = first))
    a_split = unmatricize(dename(na), blocked_axes...)
    names_split = Any[tuple.(inds(na))...]
    for splitter in splitters
        fused_name, split_namedlengths = splitter
        fused_dim = findfirst(isequal(fused_name), inds(na))
        split_names = name.(split_namedlengths)
        names_split[fused_dim] = split_names
    end
    names_split = reduce((x, y) -> (x..., y...), names_split)
    return nameddims(a_split, names_split)
end

for f in [
        :factorize, :left_orth, :left_polar, :lq, :orth, :polar, :qr, :right_orth, :right_polar,
    ]
    @eval begin
        function TensorAlgebra.$f(
                a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
            )
            codomain = to_inds(a, dimnames_codomain)
            domain = to_inds(a, dimnames_domain)
            x_unnamed, y_unnamed = $f(dename(a), inds(a), codomain, domain; kwargs...)
            name_x = randname(dimnames(a, 1))
            name_y = name_x
            namedindices_x = named(last(axes(x_unnamed)), name_x)
            namedindices_y = named(first(axes(y_unnamed)), name_y)
            inds_x = (codomain..., namedindices_x)
            inds_y = (namedindices_y, domain...)
            x = nameddims(x_unnamed, inds_x)
            y = nameddims(y_unnamed, inds_y)
            return x, y
        end
        function TensorAlgebra.$f(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
            codomain = to_inds(a, dimnames_codomain)
            domain = setdiff(inds(a), codomain)
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
    codomain = to_inds(a, dimnames_codomain)
    domain = to_inds(a, dimnames_domain)
    u_unnamed, s_unnamed, v_unnamed = svd(
        dename(a), inds(a), codomain, domain; kwargs...
    )
    name_u = randname(dimnames(a, 1))
    name_v = randname(dimnames(a, 1))
    namedindices_u = named(last(axes(u_unnamed)), name_u)
    namedindices_v = named(first(axes(v_unnamed)), name_v)
    inds_u = (codomain..., namedindices_u)
    inds_s = (namedindices_u, namedindices_v)
    inds_v = (namedindices_v, domain...)
    u = nameddims(u_unnamed, inds_u)
    s = nameddims(s_unnamed, inds_s)
    v = nameddims(v_unnamed, inds_v)
    return u, s, v
end
function TensorAlgebra.svd(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return svd(
        a,
        dimnames_codomain,
        setdiff(inds(a), to_inds(a, dimnames_codomain));
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
        inds(a),
        to_inds(a, dimnames_codomain),
        to_inds(a, dimnames_domain);
        kwargs...,
    )
end
function TensorAlgebra.svdvals(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    codomain = to_inds(a, dimnames_codomain)
    domain = setdiff(inds(a), codomain)
    return svdvals(a, codomain, domain; kwargs...)
end
function LinearAlgebra.svdvals(a::AbstractNamedDimsArray, args...; kwargs...)
    return TensorAlgebra.svdvals(a, args...; kwargs...)
end

function TensorAlgebra.eigen(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_inds(a, dimnames_codomain)
    domain = to_inds(a, dimnames_domain)
    d_unnamed, v_unnamed = eigen(dename(a), inds(a), codomain, domain; kwargs...)
    name_d = randname(dimnames(a, 1))
    name_d′ = randname(name_d)
    name_v = name_d
    namedindices_d = named(last(axes(d_unnamed)), name_d)
    namedindices_d′ = named(first(axes(d_unnamed)), name_d′)
    namedindices_v = named(last(axes(v_unnamed)), name_v)
    inds_d = (namedindices_d′, namedindices_d)
    inds_v = (domain..., namedindices_v)
    d = nameddims(d_unnamed, inds_d)
    v = nameddims(v_unnamed, inds_v)
    return d, v
end
function LinearAlgebra.eigen(a::AbstractNamedDimsArray, args...; kwargs...)
    return TensorAlgebra.eigen(a, args...; kwargs...)
end

function TensorAlgebra.eigvals(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_inds(a, dimnames_codomain)
    domain = to_inds(a, dimnames_domain)
    return eigvals(dename(a), inds(a), codomain, domain; kwargs...)
end
function LinearAlgebra.eigvals(a::AbstractNamedDimsArray, args...; kwargs...)
    return TensorAlgebra.eigvals(a, args...; kwargs...)
end

function TensorAlgebra.left_null(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_inds(a, dimnames_codomain)
    domain = to_inds(a, dimnames_domain)
    n_unnamed = left_null(dename(a), inds(a), codomain, domain; kwargs...)
    name_n = randname(dimnames(a, 1))
    namedindices_n = named(last(axes(n_unnamed)), name_n)
    inds_n = (codomain..., namedindices_n)
    return nameddims(n_unnamed, inds_n)
end
function TensorAlgebra.left_null(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    codomain = to_inds(a, dimnames_codomain)
    domain = setdiff(inds(a), codomain)
    return left_null(a, codomain, domain; kwargs...)
end

function TensorAlgebra.right_null(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_inds(a, dimnames_codomain)
    domain = to_inds(a, dimnames_domain)
    n_unnamed = right_null(dename(a), inds(a), codomain, domain; kwargs...)
    name_n = randname(dimnames(a, 1))
    namedindices_n = named(first(axes(n_unnamed)), name_n)
    inds_n = (namedindices_n, domain...)
    return nameddims(n_unnamed, inds_n)
end
function TensorAlgebra.right_null(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    codomain = to_inds(a, dimnames_codomain)
    domain = setdiff(inds(a), codomain)
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
            codomain = to_inds(a, dimnames_codomain)
            domain = to_inds(a, dimnames_domain)
            fa_unnamed = TensorAlgebra.$f(
                dename(a), inds(a), codomain, domain; kwargs...
            )
            return nameddims(fa_unnamed, (codomain..., domain...))
        end
    end
end
