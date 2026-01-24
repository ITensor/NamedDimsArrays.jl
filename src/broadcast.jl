using Base.Broadcast: Broadcast as BC, Broadcasted, broadcast_shape, broadcasted,
    check_broadcast_shape, combine_axes
## using ..NamedDimsArrays: NamedDimsArrays, AbstractNamedDimsArray,
##     AbstractNamedUnitRange, LittleSet, dename, denamed, getperm, inds, name, named,
##     nameddimsconstructorof
import TensorAlgebra as TA

abstract type AbstractNamedDimsArrayStyle{N} <: BC.AbstractArrayStyle{N} end

struct NamedDimsArrayStyle{N, NDA} <: AbstractNamedDimsArrayStyle{N} end
NamedDimsArrayStyle(::Val{N}) where {N} = NamedDimsArrayStyle{N, NamedDimsArray}()
NamedDimsArrayStyle{M}(::Val{N}) where {M, N} = NamedDimsArrayStyle{N, NamedDimsArray}()
NamedDimsArrayStyle{M, NDA}(::Val{N}) where {M, N, NDA} = NamedDimsArrayStyle{N, NDA}()

function nameddimstype(style::NamedDimsArrayStyle{<:Any, NDA}) where {NDA}
    return NDA
end

function BC.BroadcastStyle(arraytype::Type{<:AbstractNamedDimsArray})
    return NamedDimsArrayStyle{ndims(arraytype), nameddimsconstructorof(arraytype)}()
end

function BC.combine_axes(
        a1::AbstractNamedDimsArray, a_rest::AbstractNamedDimsArray...
    )
    return broadcast_shape(axes(a1), combine_axes(a_rest...))
end
function BC.combine_axes(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
    return broadcast_shape(axes(a1), axes(a2))
end
BC.combine_axes(a::AbstractNamedDimsArray) = axes(a)

function BC.broadcast_shape(
        ax1::LittleSet, ax2::LittleSet, ax_rest::LittleSet...
    )
    return broadcast_shape(broadcast_shape(ax1, ax2), ax_rest...)
end

function BC.broadcast_shape(ax1::LittleSet, ax2::LittleSet)
    return promote_shape(ax1, ax2)
end

# Handle scalar values.
function BC.broadcast_shape(ax1::Tuple{}, ax2::LittleSet)
    return ax2
end
function BC.broadcast_shape(ax1::LittleSet, ax2::Tuple{})
    return ax1
end

function Base.promote_shape(ax1::LittleSet, ax2::LittleSet)
    return LittleSet(set_promote_shape(Tuple(ax1), Tuple(ax2)))
end

function set_promote_shape(
        ax1::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange, N}},
        ax2::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange, N}},
    ) where {N}
    perm = getperm(ax2, ax1)
    ax2_aligned = map(i -> ax2[i], perm)
    ax_promoted = promote_shape(denamed.(ax1), denamed.(ax2_aligned))
    return named.(ax_promoted, name.(ax1))
end

# Handle operations like `randn() + randn(2, 2)[i, j]``.
# TODO: Decide if this should be a general definition for `AbstractNamedDimsArray`,
# or just for `AbstractITensor`.
function set_promote_shape(
        ax1::Tuple{}, ax2::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
    )
    return ax2
end

# Handle operations like `randn(2, 2)[i, j] + randn()`.
# TODO: Decide if this should be a general definition for `AbstractNamedDimsArray`,
# or just for `AbstractITensor`.
function set_promote_shape(
        ax1::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}, ax2::Tuple{}
    )
    return ax1
end

function BC.check_broadcast_shape(ax1::LittleSet, ax2::LittleSet)
    return set_check_broadcast_shape(Tuple(ax1), Tuple(ax2))
end

function set_check_broadcast_shape(
        ax1::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange, N}},
        ax2::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange, N}},
    ) where {N}
    perm = getperm(ax2, ax1)
    ax2_aligned = map(i -> ax2[i], perm)
    check_broadcast_shape(denamed.(ax1), denamed.(ax2_aligned))
    return nothing
end
set_check_broadcast_shape(ax1::Tuple{}, ax2::Tuple{}) = nothing

broadcasted_denamed(x::Number, inds) = x
broadcasted_denamed(a::AbstractArray, inds) = denamed(a, inds)
function broadcasted_denamed(bc::Broadcasted, inds)
    return broadcasted(bc.f, Base.Fix2(broadcasted_denamed, inds).(bc.args)...)
end

function Base.similar(bc::Broadcasted{<:AbstractNamedDimsArrayStyle}, elt::Type, ax)
    inds_a = name.(ax)
    bc_denamed = broadcasted_denamed(bc, inds_a)
    a_denamed = similar(bc_denamed, elt)
    return nameddimstype(bc.style)(a_denamed, inds_a)
end

## Base.axes(bc::Broadcasted) = name.(axes(bc))
function Base.copy(bc::Broadcasted{<:AbstractNamedDimsArrayStyle})
    # We could use:
    # ```julia
    # elt = combine_eltypes(bc.f, bc.args)
    # copyto!(similar(bc, elt), bc)
    # ```
    # but `combine_eltypes` is based on type inference, which might fail.
    # Calling broadcasted on the denamed arrays reuses the code logic in
    # Base.Broadcast for handling cases where type inference fails by determining
    # the output element type at runtime with widening.
    inds_dest = axes(bc)
    bc_denamed = broadcasted_denamed(bc, inds_dest)
    dest_denamed = copy(bc_denamed)
    return nameddimstype(bc.style)(dest_denamed, inds_dest)
end

function Base.copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractNamedDimsArrayStyle})
    dest_denamed = denamed(dest)
    inds_dest = axes(dest)
    bc_denamed = broadcasted_denamed(bc, inds_dest)
    copyto!(dest_denamed, bc_denamed)
    return dest
end
