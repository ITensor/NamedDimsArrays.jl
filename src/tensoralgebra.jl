import LinearAlgebra as LA
import TensorAlgebra as TA
using TupleTools: TupleTools

# This layer is used to define derivative rules (to skip differentiating `setdiff`).
inds_setdiff(s1, s2) = setdiff(s1, s2)

function TA.add!(
        dest::AbstractNamedDimsArray, src::AbstractNamedDimsArray, α::Number, β::Number
    )
    TA.add!(denamed(dest), denamed(src, axes(dest)), α, β)
    return dest
end

Base.:*(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray) = mul_nameddims(a1, a2)
function mul_nameddims(a1::AbstractArray, a2::AbstractArray)
    a_dest, inds_dest = TA.contract(
        denamed(a1), axes(a1), denamed(a2), axes(a2)
    )
    nameddimstype = combine_nameddimsconstructors(
        nameddimsconstructorof(a1), nameddimsconstructorof(a2)
    )
    return nameddimstype(a_dest, inds_dest)
end

# Left associative fold/reduction.
# Circumvent Base definitions:
# ```julia
# *(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
# *(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
# ```
# that optimize matrix multiplication sequence.
function Base.:*(
        a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray,
        a3::AbstractNamedDimsArray, a_rest::AbstractNamedDimsArray...,
    )
    return mul_nameddims(a1, a2, a3, a_rest...)
end
function mul_nameddims(
        a1::AbstractArray, a2::AbstractArray,
        a3::AbstractArray, a_rest::AbstractArray...,
    )
    return *(*(a1, a2), a3, a_rest...)
end

function LinearAlgebra.mul!(
        a_dest::AbstractNamedDimsArray,
        a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray,
        α::Number, β::Number,
    )
    return mul!_nameddims(a_dest, a1, a2, α, β)
end
function mul!_nameddims(
        a_dest::AbstractArray,
        a1::AbstractArray, a2::AbstractArray,
        α::Number, β::Number,
    )
    TA.contractadd!(
        denamed(a_dest), axes(a_dest), denamed(a1), axes(a1), denamed(a2), axes(a2), α, β
    )
    return a_dest
end

function LinearAlgebra.mul!(
        a_dest::AbstractNamedDimsArray,
        a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray,
    )
    return mul!_nameddims(a_dest, a1, a2)
end
function mul!_nameddims(
        a_dest::AbstractArray,
        a1::AbstractArray, a2::AbstractArray,
    )
    TA.contract!(denamed(a_dest), axes(a_dest), denamed(a1), axes(a1), denamed(a2), axes(a2))
    return a_dest
end

function TA.blockedperm(na::AbstractNamedDimsArray, nameddim_blocks::Tuple...)
    return blockedperm_nameddims(na, nameddim_blocks...)
end
function blockedperm_nameddims(na::AbstractArray, nameddim_blocks::Tuple...)
    dimname_blocks = map(group -> to_axes(na, group), nameddim_blocks)
    inds_a = axes(na)
    perms = map(dimname_blocks) do dimname_block
        return TA.BaseExtensions.indexin(dimname_block, inds_a)
    end
    return TA.permmortar(perms)
end

# i, j, k, l = named.((2, 2, 2, 2), ("i", "j", "k", "l"))
# a = randn(i, j, k, l)
# matricize(a, (i, k) => "a")
# matricize(a, (i, k) => "a", (j, l) => "b")
# TODO: Rewrite in terms of `matricize(a, .., (1, 3))` interface.
function TA.matricize(a::AbstractNamedDimsArray, fusions::Vararg{Pair, 2})
    return matricize_nameddims(a, fusions...)
end
function matricize_nameddims(na::AbstractArray, fusions::Vararg{Pair, 2})
    inds_fuse = map(group -> to_axes(na, group), first.(fusions))
    inds_fused = last.(fusions)
    if sum(length, inds_fuse) < ndims(na)
        # Not all names are specified
        inds_unspecified = inds_setdiff(axes(na), inds_fuse...)
        inds_fuse = vcat(
            tuple.(inds_unspecified), collect(inds_fuse)
        )
        inds_fused = vcat(
            inds_unspecified, collect(inds_fused)
        )
    end
    perm = TA.blockedperm(na, inds_fuse...)
    a_fused = TA.matricize(denamed(na), perm)
    return nameddims(a_fused, inds_fused)
end

function TA.unmatricize(na::AbstractNamedDimsArray, splitters::Vararg{Pair, 2})
    return unmatricize_nameddims(na, splitters...)
end
function unmatricize_nameddims(na::AbstractArray, splitters::Vararg{Pair, 2})
    splitters = to_axes(na, first.(splitters)) .=> last.(splitters)
    split_namedlengths = last.(splitters)
    splitters_denamed = map(splitters) do splitter
        fused_name, split_namedlengths = splitter
        fused_dim = findfirst(isequal(fused_name), axes(na))
        split_lengths = denamed.(split_namedlengths)
        return fused_dim => split_lengths
    end
    blocked_axes = last.(TupleTools.sort(splitters_denamed; by = first))
    a_split = TA.unmatricize(denamed(na), blocked_axes...)
    names_split = Any[tuple.(axes(na))...]
    for splitter in splitters
        fused_name, split_namedlengths = splitter
        fused_dim = findfirst(isequal(fused_name), axes(na))
        split_names = name.(split_namedlengths)
        names_split[fused_dim] = split_names
    end
    names_split = reduce((x, y) -> (x..., y...), names_split)
    return nameddims(a_split, names_split)
end

for f in [
        :factorize, :left_orth, :left_polar, :lq, :orth, :polar, :qr, :right_orth,
        :right_polar,
    ]
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function TA.$f(
                a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
            )
            return $f_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractArray, dimnames_codomain, dimnames_domain; kwargs...
            )
            codomain = to_axes(a, dimnames_codomain)
            domain = to_axes(a, dimnames_domain)
            x_denamed, y_denamed = TA.$f(denamed(a), axes(a), codomain, domain; kwargs...)
            name_x = randname(dimnames(a, 1))
            name_y = name_x
            namedindices_x = named(last(axes(x_denamed)), name_x)
            namedindices_y = named(first(axes(y_denamed)), name_y)
            inds_x = (codomain..., namedindices_x)
            inds_y = (namedindices_y, domain...)
            x = nameddims(x_denamed, inds_x)
            y = nameddims(y_denamed, inds_y)
            return x, y
        end
        function TA.$f(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
            return $f_nameddims(a, dimnames_codomain; kwargs...)
        end
        function $f_nameddims(a::AbstractArray, dimnames_codomain; kwargs...)
            codomain = to_axes(a, dimnames_codomain)
            domain = inds_setdiff(axes(a), codomain)
            return TA.$f(a, codomain, domain; kwargs...)
        end
    end
end

# Overload LinearAlgebra functions where relevant.
function LA.qr(a::AbstractNamedDimsArray, args...; kwargs...)
    return TA.qr(a, args...; kwargs...)
end
function LA.lq(a::AbstractNamedDimsArray, args...; kwargs...)
    return TA.lq(a, args...; kwargs...)
end
function LA.factorize(a::AbstractNamedDimsArray, args...; kwargs...)
    return TA.factorize(a, args...; kwargs...)
end

#
# Non-binary factorizations.
#

function TA.svd(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    return svd_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function svd_nameddims(
        a::AbstractArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_axes(a, dimnames_codomain)
    domain = to_axes(a, dimnames_domain)
    u_denamed, s_denamed, v_denamed = TA.svd(
        denamed(a), axes(a), codomain, domain; kwargs...
    )
    name_u = randname(dimnames(a, 1))
    name_v = randname(dimnames(a, 1))
    namedindices_u = named(last(axes(u_denamed)), name_u)
    namedindices_v = named(first(axes(v_denamed)), name_v)
    inds_u = (codomain..., namedindices_u)
    inds_s = (namedindices_u, namedindices_v)
    inds_v = (namedindices_v, domain...)
    u = nameddims(u_denamed, inds_u)
    s = nameddims(s_denamed, inds_s)
    v = nameddims(v_denamed, inds_v)
    return u, s, v
end
function TA.svd(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return svd_nameddims(a, dimnames_codomain; kwargs...)
end
function svd_nameddims(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return TA.svd(
        a,
        dimnames_codomain,
        inds_setdiff(axes(a), to_axes(a, dimnames_codomain));
        kwargs...,
    )
end
function LA.svd(a::AbstractNamedDimsArray, args...; kwargs...)
    return TA.svd(a, args...; kwargs...)
end

function TA.svdvals(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    return svdvals_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function svdvals_nameddims(
        a::AbstractArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    return TA.svdvals(
        denamed(a),
        axes(a),
        to_axes(a, dimnames_codomain),
        to_axes(a, dimnames_domain);
        kwargs...,
    )
end

function TA.svdvals(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return svdvals_nameddims(a, dimnames_codomain; kwargs...)
end
function svdvals_nameddims(a::AbstractArray, dimnames_codomain; kwargs...)
    codomain = to_axes(a, dimnames_codomain)
    domain = inds_setdiff(axes(a), codomain)
    return TA.svdvals(a, codomain, domain; kwargs...)
end

function LA.svdvals(a::AbstractNamedDimsArray, args...; kwargs...)
    return TA.svdvals(a, args...; kwargs...)
end

function TA.eigen(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    return eigen_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function eigen_nameddims(
        a::AbstractArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_axes(a, dimnames_codomain)
    domain = to_axes(a, dimnames_domain)
    d_denamed, v_denamed = TA.eigen(denamed(a), axes(a), codomain, domain; kwargs...)
    name_d = randname(dimnames(a, 1))
    name_d′ = randname(name_d)
    name_v = name_d
    namedindices_d = named(last(axes(d_denamed)), name_d)
    namedindices_d′ = named(first(axes(d_denamed)), name_d′)
    namedindices_v = named(last(axes(v_denamed)), name_v)
    inds_d = (namedindices_d′, namedindices_d)
    inds_v = (domain..., namedindices_v)
    d = nameddims(d_denamed, inds_d)
    v = nameddims(v_denamed, inds_v)
    return d, v
end

function LA.eigen(a::AbstractNamedDimsArray, args...; kwargs...)
    return TA.eigen(a, args...; kwargs...)
end

function TA.eigvals(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    return eigvals_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function eigvals_nameddims(
        a::AbstractArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_axes(a, dimnames_codomain)
    domain = to_axes(a, dimnames_domain)
    return TA.eigvals(denamed(a), axes(a), codomain, domain; kwargs...)
end

function LA.eigvals(a::AbstractNamedDimsArray, args...; kwargs...)
    return TA.eigvals(a, args...; kwargs...)
end

function TA.left_null(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    return left_null_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function left_null_nameddims(
        a::AbstractArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_axes(a, dimnames_codomain)
    domain = to_axes(a, dimnames_domain)
    n_denamed = TA.left_null(denamed(a), axes(a), codomain, domain; kwargs...)
    name_n = randname(dimnames(a, 1))
    namedindices_n = named(last(axes(n_denamed)), name_n)
    inds_n = (codomain..., namedindices_n)
    return nameddims(n_denamed, inds_n)
end

function TA.left_null(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return left_null_nameddims(a, dimnames_codomain; kwargs...)
end
function left_null_nameddims(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    codomain = to_axes(a, dimnames_codomain)
    domain = inds_setdiff(axes(a), codomain)
    return TA.left_null(a, codomain, domain; kwargs...)
end

function TA.right_null(
        a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    return right_null_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
end
function right_null_nameddims(
        a::AbstractArray, dimnames_codomain, dimnames_domain; kwargs...
    )
    codomain = to_axes(a, dimnames_codomain)
    domain = to_axes(a, dimnames_domain)
    n_denamed = TA.right_null(denamed(a), axes(a), codomain, domain; kwargs...)
    name_n = randname(dimnames(a, 1))
    namedindices_n = named(first(axes(n_denamed)), name_n)
    inds_n = (namedindices_n, domain...)
    return nameddims(n_denamed, inds_n)
end

function TA.right_null(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return right_null_nameddims(a, dimnames_codomain; kwargs...)
end
function right_null_nameddims(a::AbstractArray, dimnames_codomain; kwargs...)
    codomain = to_axes(a, dimnames_codomain)
    domain = inds_setdiff(axes(a), codomain)
    return TA.right_null(a, codomain, domain; kwargs...)
end

const MATRIX_FUNCTIONS = [
    :exp, :cis, :log, :sqrt, :cbrt, :cos, :sin, :tan, :csc, :sec, :cot, :cosh, :sinh, :tanh,
    :csch, :sech, :coth, :acos, :asin, :atan, :acsc, :asec, :acot, :acosh, :asinh, :atanh,
    :acsch, :asech, :acoth,
]

for f in MATRIX_FUNCTIONS
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function Base.$f(
                a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; kwargs...
            )
            return $f_nameddims(a, dimnames_codomain, dimnames_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractArray, dimnames_codomain, dimnames_domain; kwargs...
            )
            codomain = to_axes(a, dimnames_codomain)
            domain = to_axes(a, dimnames_domain)
            fa_denamed = TA.$f(
                denamed(a), axes(a), codomain, domain; kwargs...
            )
            return nameddims(fa_denamed, (codomain..., domain...))
        end
    end
end
