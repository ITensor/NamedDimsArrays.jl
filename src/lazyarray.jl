# TODO: Move this to TensorAlgebra.jl.
using TermInterface: TermInterface as TI, operation, arguments

# Primitive constructors for lazy array linear algebra operations.
+ₗ(a::AbstractArray, b::AbstractArray) = AddArray(a, b)
*ₗ(a::AbstractArray, b::AbstractArray) = MulArray(a, b)
*ₗ(α::Number, a::AbstractArray) = ScaledArray(α, a)
conjed(a::AbstractArray) = ConjArray(a)
conjed(a::AbstractArray{<:Real}) = a

# Generic logic for lazy array linear algebra operations.
function +ₗ(a::AbstractArray, b::AbstractArray, c::AbstractArray, xs::AbstractArray...)
    return Base.afoldl(+ₗ, +ₗ(+ₗ(a, b), c), xs...)
end
-ₗ(a::AbstractArray, b::AbstractArray) = a +ₗ (-b)
function *ₗ(a::AbstractArray, b::AbstractArray, c::AbstractArray, xs::AbstractArray...)
    return Base.afoldl(*ₗ, *ₗ(*ₗ(a, b), c), xs...)
end
*ₗ(a::AbstractArray, b::Number) = b *ₗ a
\ₗ(a::Number, b::AbstractArray) = inv(a) *ₗ b
/ₗ(a::AbstractArray, b::Number) = a *ₗ inv(b)
+ₗ(a::AbstractArray) = a
-ₗ(a::AbstractArray) = -1 *ₗ a

abstract type AbstractLazyArray{T, N} <: AbstractArray{T, N} end

# For lazy arrays, define Base methods in terms of lazy operations.
Base.:(+)(a::AbstractLazyArray, b::AbstractLazyArray) = a +ₗ b
Base.:(+)(a::AbstractLazyArray, b::AbstractArray) = a +ₗ b
Base.:(+)(a::AbstractArray, b::AbstractLazyArray) = a +ₗ b
Base.:(*)(α::Number, a::AbstractLazyArray) = α *ₗ a
Base.:(*)(a::AbstractLazyArray, α::Number) = a *ₗ α
Base.:(\)(α::Number, a::AbstractLazyArray) = α \ₗ a
Base.:(/)(a::AbstractLazyArray, α::Number) = a /ₗ α
Base.:(-)(a::AbstractLazyArray) = -ₗ(a)
Base.conj(a::AbstractLazyArray) = conjed(a)
function Base.similar(a::AbstractLazyArray, elt::Type)
    return similar(a, elt, axes(a))
end
function Base.similar(a::AbstractLazyArray, elt::Type, ax)
    return error("Not implemented.")
end

# Interface for materializing a lazy array.
Base.copy(a::AbstractLazyArray) = copyto!(similar(a), a)
function Base.copyto!(dest::AbstractArray, src::AbstractLazyArray)
    return error("Not implemented.")
end

function Base.show(io::IO, a::AbstractLazyArray)
    print(io, operation(a), "(", join(arguments(a), ", "), ")")
    return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::AbstractLazyArray)
    summary(io, a)
    println(io, ":")
    show(io, a)
    return nothing
end

scaled_eltype(coeff, a::AbstractArray) = Base.promote_op(*, typeof(coeff), eltype(a))

struct ScaledArray{T, N, C <: Number, TP, P <: AbstractArray{TP, N}} <:
    AbstractLazyArray{T, N}
    coeff::C
    parent::P
    function ScaledArray{T}(coeff::Number, a::AbstractArray) where {T}
        @assert scaled_eltype(coeff, a) == T begin
            lazy"target type $T cannot hold products of " *
                lazy"$(typeof(coeff)) and $(eltype(a)) objects"
        end
        return new{T, ndims(a), typeof(coeff), eltype(a), typeof(a)}(coeff, a)
    end
end
function ScaledArray(coeff::Number, a::AbstractArray)
    return ScaledArray{scaled_eltype(coeff, a)}(coeff, a)
end
Base.eltype(a::ScaledArray) = scaled_eltype(a.coeff, a.parent)
Base.ndims(a::ScaledArray) = ndims(a.parent)

Base.similar(a::ScaledArray) = similar(a.parent)
Base.similar(a::ScaledArray, elt::Type) = similar(a.parent, elt)
Base.similar(a::ScaledArray, ax) = similar(a.parent, ax)
Base.similar(a::ScaledArray, elt::Type, ax) = similar(a.parent, elt, ax)

using TensorAlgebra: add!
function Base.copyto!(dest::AbstractArray, src::ScaledArray)
    add!(dest, src.parent, src.coeff, false)
    return dest
end
Base.axes(a::ScaledArray) = axes(a.parent)
Base.size(a::ScaledArray) = size(a.parent)
TI.iscall(::ScaledArray) = true
TI.operation(::ScaledArray) = *
TI.arguments(a::ScaledArray) = (a.coeff, a.parent)

*ₗ(α::Number, a::ScaledArray) = (α * a.coeff) *ₗ a.parent
conjed(a::ScaledArray) = conj(a.coeff) *ₗ conjed(a.parent)

struct ConjArray{T, N, P <: AbstractArray{T, N}} <:
    AbstractLazyArray{T, N}
    parent::P
end
Base.eltype(a::ConjArray) = eltype(a.parent)
Base.ndims(a::ConjArray) = ndims(a.parent)

Base.similar(a::ConjArray, elt::Type) = similar(a.parent, elt)
Base.similar(a::ConjArray, elt::Type, ax) = similar(a.parent, elt, ax)

using StridedViews: StridedViews, StridedView, isstrided
StridedViews.isstrided(a::ConjArray) = isstrided(a.parent)
StridedViews.StridedView(a::ConjArray) = conj(StridedView(a.parent))

using TensorAlgebra: add!
function Base.copyto!(dest::AbstractArray, src::ConjArray)
    add!(dest, src, true, false)
    return dest
end
Base.axes(a::ConjArray) = axes(a.parent)
Base.size(a::ConjArray) = size(a.parent)
TI.iscall(::ConjArray) = true
TI.operation(::ConjArray) = conj
TI.arguments(a::ConjArray) = (a.parent,)

Base.conj(a::ConjArray) = a.parent
