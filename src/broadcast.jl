# Broadcasting NamedDimsArrays. Wrap in a module to avoid name conflicts with
# FunctionImplementations Style definitions.
module Broadcast

using Base.Broadcast: Broadcast as BC, Broadcasted, broadcast_shape, broadcasted,
    check_broadcast_shape, combine_axes
using MapBroadcast: MapBroadcast, Mapped, mapped, tile
using ..NamedDimsArrays: NamedDimsArrays, AbstractNamedDimsArray,
    AbstractNamedUnitRange, NaiveOrderedSet, dename, denamed, getperm, name, named,
    nameddimsconstructorof

abstract type AbstractNamedDimsArrayStyle{N} <: BC.AbstractArrayStyle{N} end

struct NamedDimsArrayStyle{N, NDA} <: AbstractNamedDimsArrayStyle{N} end
NamedDimsArrayStyle(::Val{N}) where {N} = NamedDimsArrayStyle{N, NamedDimsArray}()
NamedDimsArrayStyle{M}(::Val{N}) where {M, N} = NamedDimsArrayStyle{N, NamedDimsArray}()
NamedDimsArrayStyle{M, NDA}(::Val{N}) where {M, N, NDA} = NamedDimsArrayStyle{N, NDA}()

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
        ax1::NaiveOrderedSet, ax2::NaiveOrderedSet, ax_rest::NaiveOrderedSet...
    )
    return broadcast_shape(broadcast_shape(ax1, ax2), ax_rest...)
end

function BC.broadcast_shape(ax1::NaiveOrderedSet, ax2::NaiveOrderedSet)
    return promote_shape(ax1, ax2)
end

# Handle scalar values.
function BC.broadcast_shape(ax1::Tuple{}, ax2::NaiveOrderedSet)
    return ax2
end
function BC.broadcast_shape(ax1::NaiveOrderedSet, ax2::Tuple{})
    return ax1
end

function Base.promote_shape(ax1::NaiveOrderedSet, ax2::NaiveOrderedSet)
    return NaiveOrderedSet(set_promote_shape(Tuple(ax1), Tuple(ax2)))
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

# Handle operations like `ITensor() + ITensor(i, j)`.
# TODO: Decide if this should be a general definition for `AbstractNamedDimsArray`,
# or just for `AbstractITensor`.
function set_promote_shape(
        ax1::Tuple{}, ax2::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}
    )
    return ax2
end

# Handle operations like `ITensor(i, j) + ITensor()`.
# TODO: Decide if this should be a general definition for `AbstractNamedDimsArray`,
# or just for `AbstractITensor`.
function set_promote_shape(
        ax1::Tuple{AbstractNamedUnitRange, Vararg{AbstractNamedUnitRange}}, ax2::Tuple{}
    )
    return ax1
end

function BC.check_broadcast_shape(ax1::NaiveOrderedSet, ax2::NaiveOrderedSet)
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

# Dename and lazily permute the arguments using the reference
# dimension names.
# TODO: Make a version that gets the inds from `m`.
function NamedDimsArrays.denamed(m::Mapped, inds)
    return mapped(m.f, map(arg -> denamed(arg, inds), m.args)...)
end

function nameddimstype(style::NamedDimsArrayStyle{<:Any, NDA}) where {NDA}
    return NDA
end

using FillArrays: Fill

function MapBroadcast.tile(a::AbstractNamedDimsArray, ax)
    axes(a) == ax && return a
    !iszero(ndims(a)) && return error("Not implemented.")
    return nameddimsconstructorof(a)(Fill(a[], denamed.(Tuple(ax))), name.(ax))
end

function Base.similar(bc::Broadcasted{<:AbstractNamedDimsArrayStyle}, elt::Type, ax)
    inds = name.(ax)
    m′ = denamed(Mapped(bc), inds)
    # TODO: Store the wrapper type in `AbstractNamedDimsArrayStyle` and use that
    # wrapper type rather than the generic `nameddims` constructor, which
    # can lose information.
    # Call it as `nameddimstype(bc.style)`.
    return nameddimstype(bc.style)(
        similar(m′, elt, denamed.(Tuple(ax))), inds
    )
end

function Base.copyto!(dest::AbstractArray, bc::Broadcasted{<:AbstractNamedDimsArrayStyle})
    return copyto!(dest, Mapped(bc))
end

end
