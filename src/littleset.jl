using Base.Broadcast: AbstractArrayStyle, Broadcasted, Style

struct LittleSet{Values}
    values::Values
end
Base.values(s::LittleSet) = s.values
Base.Tuple(s::LittleSet) = Tuple(values(s))
Base.eltype(s::LittleSet) = eltype(values(s))
Base.length(s::LittleSet) = length(values(s))
Base.axes(s::LittleSet) = axes(values(s))
Base.keys(s::LittleSet) = Base.OneTo(length(s))
Base.:(==)(s1::LittleSet, s2::LittleSet) = issetequal(values(s1), values(s2))
Base.iterate(s::LittleSet, args...) = iterate(values(s), args...)
Base.getindex(s::LittleSet, I::Int) = values(s)[I]
# TODO: Required in Julia 1.10, delete when we drop support for that.
Base.getindex(s::LittleSet, I::CartesianIndex{1}) = values(s)[I]
Base.get(s::LittleSet, I::Integer, default) = get(values(s), I, default)
Base.invperm(s::LittleSet) = LittleSet(invperm(values(s)))
Base.Broadcast._axes(::Broadcasted, axes::LittleSet) = axes
Base.Broadcast.BroadcastStyle(::Type{<:LittleSet}) = Style{LittleSet}()
Base.Broadcast.BroadcastStyle(::Style{Tuple}, ::Style{LittleSet}) = Style{Tuple}()
Base.Broadcast.BroadcastStyle(s1::AbstractArrayStyle{0}, s2::Style{LittleSet}) = s2
Base.Broadcast.BroadcastStyle(s1::AbstractArrayStyle, s2::Style{LittleSet}) = s1
Base.Broadcast.broadcastable(s::LittleSet) = s
Base.to_shape(s::LittleSet) = s

# Needed for functionality such as `CartesianIndices(::AbstractNamedDimsArray)`,
# `pairs(::AbstractNamedDimsArray)`, etc.
Base.CartesianIndices(s::LittleSet) = CartesianIndices(values(s))

function Base.copy(
        bc::Broadcasted{Style{LittleSet}, <:Any, <:Any, <:Tuple{<:LittleSet}}
    )
    return LittleSet(bc.f.(values(only(bc.args))))
end
# Multiple arguments not supported.
function Base.copy(bc::Broadcasted{Style{LittleSet}})
    return error("This broadcasting expression of `LittleSet` is not supported.")
end
function Base.map(f::Function, s::LittleSet)
    return LittleSet(map(f, values(s)))
end
function Base.replace(f::Union{Function, Type}, s::LittleSet; kwargs...)
    return LittleSet(replace(f, values(s); kwargs...))
end
function Base.replace(s::LittleSet, replacements::Pair...; kwargs...)
    return LittleSet(replace(values(s), replacements...; kwargs...))
end
