import LinearAlgebra as LA
import TensorAlgebra as TA
using TupleTools: TupleTools

# This layer is used to define derivative rules (to skip differentiating `setdiff`).
dimnames_setdiff(s1, s2) = setdiff(s1, s2)

function TA.add!(
        dest::AbstractNamedDimsArray, src::AbstractNamedDimsArray, α::Number, β::Number
    )
    TA.add!(denamed(dest), denamed(src, dimnames(dest)), α, β)
    return dest
end

Base.:*(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray) = mul_nameddims(a1, a2)
function mul_nameddims(a1::AbstractArray, a2::AbstractArray)
    a_dest, dimnames_dest = TA.contract(
        denamed(a1), dimnames(a1), denamed(a2), dimnames(a2)
    )
    nameddimstype = combine_nameddimsconstructors(
        nameddimsconstructorof(a1), nameddimsconstructorof(a2)
    )
    return nameddimstype(a_dest, dimnames_dest)
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
        denamed(a_dest), dimnames(a_dest),
        denamed(a1), dimnames(a1),
        denamed(a2), dimnames(a2),
        α, β,
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
    TA.contract!(
        denamed(a_dest), dimnames(a_dest),
        denamed(a1), dimnames(a1),
        denamed(a2), dimnames(a2),
    )
    return a_dest
end

function TA.blockedperm(na::AbstractNamedDimsArray, nameddim_blocks::Tuple...)
    return blockedperm_nameddims(na, nameddim_blocks...)
end
function blockedperm_nameddims(a::AbstractArray, nameddim_blocks::Tuple...)
    dimname_blocks = map(group -> name.(group), nameddim_blocks)
    dimnames_a = dimnames(a)
    perms = map(dimname_blocks) do dimname_block
        return TA.BaseExtensions.indexin(dimname_block, dimnames_a)
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
    dimnames_fuse = map(group -> name.(group), first.(fusions))
    dimnames_fused = last.(fusions)
    if sum(length, dimnames_fuse) < ndims(na)
        # Not all names are specified
        dimnames_unspecified = dimnames_setdiff(dimnames(na), dimnames_fuse...)
        dimnames_fuse = vcat(
            tuple.(dimnames_unspecified), collect(dimnames_fuse)
        )
        dimnames_fused = vcat(
            dimnames_unspecified, collect(dimnames_fused)
        )
    end
    perm = TA.blockedperm(na, dimnames_fuse...)
    a_fused = TA.matricize(denamed(na), perm)
    return nameddims(a_fused, dimnames_fused)
end

function TA.unmatricize(na::AbstractNamedDimsArray, splitters::Vararg{Pair, 2})
    return unmatricize_nameddims(na, splitters...)
end
function unmatricize_nameddims(na::AbstractArray, splitters::Vararg{Pair, 2})
    splitters = name.(first.(splitters)) .=> last.(splitters)
    split_namedlengths = last.(splitters)
    splitters_denamed = map(splitters) do splitter
        fused_name, split_namedlengths = splitter
        fused_dim = findfirst(isequal(fused_name), dimnames(na))
        split_lengths = denamed.(split_namedlengths)
        return fused_dim => split_lengths
    end
    blocked_axes = last.(TupleTools.sort(splitters_denamed; by = first))
    a_split = TA.unmatricize(denamed(na), blocked_axes...)
    names_split = Any[tuple.(dimnames(na))...]
    for splitter in splitters
        fused_name, split_namedlengths = splitter
        fused_dim = findfirst(isequal(fused_name), dimnames(na))
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
            codomain = name.(dimnames_codomain)
            domain = name.(dimnames_domain)
            x_denamed, y_denamed = TA.$f(denamed(a), dimnames(a), codomain, domain; kwargs...)
            name_x = randname(dimnames(a, 1))
            name_y = name_x
            dimnames_x = (codomain..., name_x)
            dimnames_y = (name_y, domain...)
            x = nameddims(x_denamed, dimnames_x)
            y = nameddims(y_denamed, dimnames_y)
            return x, y
        end
        function TA.$f(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
            return $f_nameddims(a, dimnames_codomain; kwargs...)
        end
        function $f_nameddims(a::AbstractArray, dimnames_codomain; kwargs...)
            codomain = name.(dimnames_codomain)
            domain = dimnames_setdiff(dimnames(a), codomain)
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
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    u_denamed, s_denamed, v_denamed = TA.svd(
        denamed(a), dimnames(a), codomain, domain; kwargs...
    )
    name_u = randname(dimnames(a, 1))
    name_v = randname(dimnames(a, 1))
    dimnames_u = (codomain..., name_u)
    dimnames_s = (name_u, name_v)
    dimnames_v = (name_v, domain...)
    u = nameddims(u_denamed, dimnames_u)
    s = nameddims(s_denamed, dimnames_s)
    v = nameddims(v_denamed, dimnames_v)
    return u, s, v
end
function TA.svd(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return svd_nameddims(a, dimnames_codomain; kwargs...)
end
function svd_nameddims(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return TA.svd(
        a,
        dimnames_codomain,
        dimnames_setdiff(dimnames(a), name.(dimnames_codomain));
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
        dimnames(a),
        name.(dimnames_codomain),
        name.(dimnames_domain);
        kwargs...,
    )
end

function TA.svdvals(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return svdvals_nameddims(a, dimnames_codomain; kwargs...)
end
function svdvals_nameddims(a::AbstractArray, dimnames_codomain; kwargs...)
    codomain = name.(dimnames_codomain)
    domain = dimnames_setdiff(dimnames(a), codomain)
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
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    d_denamed, v_denamed = TA.eigen(denamed(a), dimnames(a), codomain, domain; kwargs...)
    name_d = randname(dimnames(a, 1))
    name_d′ = randname(name_d)
    name_v = name_d
    dimnames_d = (name_d′, name_d)
    dimnames_v = (domain..., name_v)
    d = nameddims(d_denamed, dimnames_d)
    v = nameddims(v_denamed, dimnames_v)
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
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    return TA.eigvals(denamed(a), dimnames(a), codomain, domain; kwargs...)
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
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    n_denamed = TA.left_null(denamed(a), dimnames(a), codomain, domain; kwargs...)
    name_n = randname(dimnames(a, 1))
    dimnames_n = (codomain..., name_n)
    return nameddims(n_denamed, dimnames_n)
end

function TA.left_null(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return left_null_nameddims(a, dimnames_codomain; kwargs...)
end
function left_null_nameddims(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    codomain = name.(dimnames_codomain)
    domain = dimnames_setdiff(dimnames(a), codomain)
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
    codomain = name.(dimnames_codomain)
    domain = name.(dimnames_domain)
    n_denamed = TA.right_null(denamed(a), dimnames(a), codomain, domain; kwargs...)
    name_n = randname(dimnames(a, 1))
    dimnames_n = (name_n, domain...)
    return nameddims(n_denamed, dimnames_n)
end

function TA.right_null(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
    return right_null_nameddims(a, dimnames_codomain; kwargs...)
end
function right_null_nameddims(a::AbstractArray, dimnames_codomain; kwargs...)
    codomain = name.(dimnames_codomain)
    domain = dimnames_setdiff(dimnames(a), codomain)
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
            codomain = name.(dimnames_codomain)
            domain = name.(dimnames_domain)
            fa_denamed = TA.$f(
                denamed(a), dimnames(a), codomain, domain; kwargs...
            )
            return nameddims(fa_denamed, (codomain..., domain...))
        end
    end
end
