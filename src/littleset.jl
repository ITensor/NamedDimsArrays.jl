using Base.Broadcast: AbstractArrayStyle, Broadcasted, Style

struct LittleSet{Values}
    values::Values
end
Base.Tuple(s::LittleSet) = Tuple(s.values)
Base.eltype(s::LittleSet) = eltype(s.values)
Base.length(s::LittleSet) = length(s.values)
Base.axes(s::LittleSet) = axes(s.values)
Base.keys(s::LittleSet) = Base.OneTo(length(s))
Base.:(==)(s1::LittleSet, s2::LittleSet) = issetequal(s1.values, s2.values)
Base.:(==)(s1::LittleSet, s2) = s1.values == s2
Base.:(==)(s1, s2::LittleSet) = s1 == s2.values
Base.iterate(s::LittleSet, args...) = iterate(s.values, args...)
Base.getindex(s::LittleSet, I::Int) = s.values[I]
# TODO: Required in Julia 1.10, delete when we drop support for that.
Base.getindex(s::LittleSet, I::CartesianIndex{1}) = s.values[I]
Base.get(s::LittleSet, I::Integer, default) = get(s.values, I, default)
Base.invperm(s::LittleSet) = LittleSet(invperm(s.values))
# Use `_sort` to handle `Tuple` in Julia v1.10.
# TODO: Delete once support for that is dropped.
Base.sort(s::LittleSet; kwargs...) = LittleSet(_sort(s.values; kwargs...))
Base.Broadcast._axes(::Broadcasted, axes::LittleSet) = axes
Base.Broadcast.BroadcastStyle(::Type{<:LittleSet}) = Style{LittleSet}()
Base.Broadcast.BroadcastStyle(::Style{Tuple}, ::Style{LittleSet}) = Style{Tuple}()
Base.Broadcast.BroadcastStyle(s1::AbstractArrayStyle{0}, s2::Style{LittleSet}) = s2
Base.Broadcast.BroadcastStyle(s1::AbstractArrayStyle, s2::Style{LittleSet}) = s1
Base.Broadcast.broadcastable(s::LittleSet) = s
Base.to_shape(s::LittleSet) = s

# Needed for functionality such as `CartesianIndices(::AbstractNamedDimsArray)`,
# `pairs(::AbstractNamedDimsArray)`, etc.
Base.CartesianIndices(s::LittleSet) = CartesianIndices(s.values)

function Base.copy(
        bc::Broadcasted{Style{LittleSet}, <:Any, <:Any, <:Tuple{<:LittleSet}}
    )
    return LittleSet(bc.f.(only(bc.args).values))
end
# Multiple arguments not supported.
function Base.copy(bc::Broadcasted{Style{LittleSet}})
    return error("This broadcasting expression of `LittleSet` is not supported.")
end
function Base.map(f::Function, s::LittleSet)
    return LittleSet(map(f, s.values))
end
function Base.replace(f::Union{Function, Type}, s::LittleSet; kwargs...)
    return LittleSet(replace(f, s.values; kwargs...))
end
function Base.replace(s::LittleSet, replacements::Pair...; kwargs...)
    return LittleSet(replace(s.values, replacements...; kwargs...))
end
