using Base.Broadcast: AbstractArrayStyle, Broadcasted, Style

struct NaiveOrderedSet{Values}
    values::Values
end
Base.values(s::NaiveOrderedSet) = s.values
Base.Tuple(s::NaiveOrderedSet) = Tuple(values(s))
Base.eltype(s::NaiveOrderedSet) = eltype(values(s))
Base.length(s::NaiveOrderedSet) = length(values(s))
Base.axes(s::NaiveOrderedSet) = axes(values(s))
Base.keys(s::NaiveOrderedSet) = Base.OneTo(length(s))
Base.:(==)(s1::NaiveOrderedSet, s2::NaiveOrderedSet) = issetequal(values(s1), values(s2))
Base.iterate(s::NaiveOrderedSet, args...) = iterate(values(s), args...)
Base.getindex(s::NaiveOrderedSet, I::Int) = values(s)[I]
# TODO: Required in Julia 1.10, delete when we drop support for that.
Base.getindex(s::NaiveOrderedSet, I::CartesianIndex{1}) = values(s)[I]
Base.get(s::NaiveOrderedSet, I::Integer, default) = get(values(s), I, default)
Base.invperm(s::NaiveOrderedSet) = NaiveOrderedSet(invperm(values(s)))
Base.Broadcast._axes(::Broadcasted, axes::NaiveOrderedSet) = axes
Base.Broadcast.BroadcastStyle(::Type{<:NaiveOrderedSet}) = Style{NaiveOrderedSet}()
Base.Broadcast.BroadcastStyle(::Style{Tuple}, ::Style{NaiveOrderedSet}) = Style{Tuple}()
Base.Broadcast.BroadcastStyle(s1::AbstractArrayStyle{0}, s2::Style{NaiveOrderedSet}) = s2
Base.Broadcast.BroadcastStyle(s1::AbstractArrayStyle, s2::Style{NaiveOrderedSet}) = s1
Base.Broadcast.broadcastable(s::NaiveOrderedSet) = s
Base.to_shape(s::NaiveOrderedSet) = s

function Base.copy(
        bc::Broadcasted{Style{NaiveOrderedSet}, <:Any, <:Any, <:Tuple{<:NaiveOrderedSet}}
    )
    return NaiveOrderedSet(bc.f.(values(only(bc.args))))
end
# Multiple arguments not supported.
function Base.copy(bc::Broadcasted{Style{NaiveOrderedSet}})
    return error("This broadcasting expression of `NaiveOrderedSet` is not supported.")
end
function Base.map(f::Function, s::NaiveOrderedSet)
    return NaiveOrderedSet(map(f, values(s)))
end
function Base.replace(f::Union{Function, Type}, s::NaiveOrderedSet; kwargs...)
    return NaiveOrderedSet(replace(f, values(s); kwargs...))
end
function Base.replace(s::NaiveOrderedSet, replacements::Pair...; kwargs...)
    return NaiveOrderedSet(replace(values(s), replacements...; kwargs...))
end
