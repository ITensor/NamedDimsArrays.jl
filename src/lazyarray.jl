# TODO: Move this file to TensorAlgebra.jl.

import LinearAlgebra as LA
import TensorAlgebra as TA
import TermInterface as TI

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
const AbstractLazyVector{T} = AbstractLazyArray{T, 1}
const AbstractLazyMatrix{T} = AbstractLazyArray{T, 2}
const AbstractLazyVecOrMat{T} = Union{AbstractLazyVector{T}, AbstractLazyMatrix{T}}

# For lazy arrays, define Base methods in terms of lazy operations.
Base.:(+)(a::AbstractLazyArray, b::AbstractLazyArray) = a +ₗ b
Base.:(+)(a::AbstractLazyArray, b::AbstractArray) = a +ₗ b
Base.:(+)(a::AbstractArray, b::AbstractLazyArray) = a +ₗ b
Base.:(*)(α::Number, a::AbstractLazyArray) = α *ₗ a
Base.:(*)(a::AbstractLazyArray, α::Number) = a *ₗ α
Base.:(*)(a::AbstractLazyMatrix, b::AbstractLazyVecOrMat) = a *ₗ b
Base.:(*)(a::AbstractLazyMatrix, b::AbstractVecOrMat) = a *ₗ b
Base.:(*)(a::AbstractMatrix, b::AbstractLazyVecOrMat) = a *ₗ b
Base.:(\)(α::Number, a::AbstractLazyArray) = α \ₗ a
Base.:(/)(a::AbstractLazyArray, α::Number) = a /ₗ α
Base.:(-)(a::AbstractLazyArray) = -ₗ(a)
Base.conj(a::AbstractLazyArray) = conjed(a)

function Base.similar(a::AbstractLazyArray, elt::Type)
    return similar(a, elt, axes(a))
end
function Base.similar(a::AbstractLazyArray, elt::Type, ax)
    # TODO: Define fallback as:
    # args = TI.arguments(a)
    # return similar(args[findfirst(Base.Fix2(isa, AbstractArray), args)], elt, ax)
    return error("Not implemented.")
end

# Interface for materializing a lazy array.
Base.copy(a::AbstractLazyArray) = copyto!(similar(a), a)
function Base.copyto!(dest::AbstractArray, src::AbstractLazyArray)
    TA.add!(dest, src, true, false)
    return dest
end

function Base.show(io::IO, a::AbstractLazyArray)
    print(io, TI.operation(a), "(", join(TI.arguments(a), ", "), ")")
    return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::AbstractLazyArray)
    summary(io, a)
    println(io, ":")
    show(io, a)
    return nothing
end

scaled_eltype(coeff, a::AbstractArray) = Base.promote_op(*, typeof(coeff), eltype(a))

struct ScaledArray{T, N, P <: AbstractArray{<:Any, N}, C <: Number} <:
    AbstractLazyArray{T, N}
    coeff::C
    parent::P
    function ScaledArray(coeff::Number, a::AbstractArray)
        T = scaled_eltype(coeff, a)
        return new{T, ndims(a), typeof(a), typeof(coeff)}(coeff, a)
    end
end
Base.axes(a::ScaledArray) = axes(a.parent)
Base.size(a::ScaledArray) = size(a.parent)

Base.similar(a::ScaledArray) = similar(a.parent)
Base.similar(a::ScaledArray, elt::Type) = similar(a.parent, elt)
Base.similar(a::ScaledArray, ax) = similar(a.parent, ax)
Base.similar(a::ScaledArray, elt::Type, ax) = similar(a.parent, elt, ax)

function TA.add!(dest::AbstractArray, src::ScaledArray, α::Number, β::Number)
    TA.add!(dest, src.parent, src.coeff * α, β)
    return dest
end
TI.iscall(::ScaledArray) = true
TI.operation(::ScaledArray) = *
TI.arguments(a::ScaledArray) = (a.coeff, a.parent)

*ₗ(α::Number, a::ScaledArray) = (α * a.coeff) *ₗ a.parent
*ₗ(a::ScaledArray, b::ScaledArray) = (a.coeff * b.coeff) *ₗ (a.parent *ₗ b.parent)
*ₗ(a::AbstractArray, b::ScaledArray) = b.coeff *ₗ (a *ₗ b.parent)
*ₗ(a::ScaledArray, b::AbstractArray) = a.coeff *ₗ (a.parent *ₗ b)
conjed(a::ScaledArray) = conj(a.coeff) *ₗ conjed(a.parent)

struct ConjArray{T, N, P <: AbstractArray{T, N}} <:
    AbstractLazyArray{T, N}
    parent::P
end
Base.axes(a::ConjArray) = axes(a.parent)
Base.size(a::ConjArray) = size(a.parent)

Base.similar(a::ConjArray, elt::Type) = similar(a.parent, elt)
Base.similar(a::ConjArray, elt::Type, ax) = similar(a.parent, elt, ax)

using StridedViews: StridedViews, StridedView, isstrided
StridedViews.isstrided(a::ConjArray) = isstrided(a.parent)
StridedViews.StridedView(a::ConjArray) = conj(StridedView(a.parent))

TI.iscall(::ConjArray) = true
TI.operation(::ConjArray) = conj
TI.arguments(a::ConjArray) = (a.parent,)

Base.conj(a::ConjArray) = a.parent

function add_eltype(args::AbstractArray{<:Any, N}...) where {N}
    return Base.promote_op(+, eltype.(args)...)
end
using Base.Broadcast: broadcasted
add_axes(args::AbstractArray{<:Any, N}...) where {N} = axes(broadcasted(+, args...))

struct AddArray{T, N, Args <: Tuple{Vararg{AbstractArray{<:Any, N}}}} <:
    AbstractLazyArray{T, N}
    args::Args
    function AddArray(args::AbstractArray{<:Any, N}...) where {N}
        T = add_eltype(args...)
        return new{T, N, typeof(args)}(args)
    end
end
const AddVector{T, Args <: Tuple{Vararg{AbstractVector}}} = AddArray{T, 1, Args}
const AddMatrix{T, Args <: Tuple{Vararg{AbstractMatrix}}} = AddArray{T, 2, Args}
const AddVecOrMat{T} = Union{AddVector{T}, AddMatrix{T}}

Base.axes(a::AddArray) = add_axes(a.args...)
Base.size(a::AddArray) = length.(axes(a))

using Base.Broadcast: broadcasted
Base.similar(a::AddArray) = similar(a, eltype(a))
Base.similar(a::AddArray, ax::Tuple) = similar(a, eltype(a), ax)
Base.similar(a::AddArray, elt::Type) = similar_add(a, elt)
function Base.similar(
        a::AddArray,
        elt::Type,
        ax::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}},
    )
    return similar_add(a, elt, ax)
end
Base.similar(a::AddArray, elt::Type, ax) = similar_add(a, elt, ax)
using Base.Broadcast: broadcasted
similar_add(a::AddArray, elt) = similar(broadcasted(+, a.args...), elt)
function similar_add(a::AddArray, elt::Type, ax)
    return similar(broadcasted(+, a.args...), elt, ax)
end

function TA.add!(dest::AbstractArray, src::AddArray, α::Number, β::Number)
    TA.add!(dest, first(src.args), α, β)
    for a in Base.tail(src.args)
        TA.add!(dest, a, α, true)
    end
    return dest
end
TI.iscall(::AddArray) = true
TI.operation(::AddArray) = +
TI.arguments(a::AddArray) = a.args

+ₗ(a::AbstractArray, b::AddArray) = AddArray((a, b.args...)...)
+ₗ(a::AddArray, b::AbstractArray) = AddArray((a.args..., b)...)
+ₗ(a::AddArray, b::AddArray) = +ₗ((a.args..., b.args...)...)
*ₗ(α::Number, a::AddArray) = +ₗ((α .*ₗ a.args)...)
*ₗ(a::AbstractArray, b::AddArray) = +ₗ((Ref(a) .*ₗ b.args)...)
*ₗ(a::AddArray, b::AbstractArray) = +ₗ((a.args .*ₗ Ref(b))...)
*ₗ(a::AddArray, b::AddArray) = +ₗ((Ref(a) .*ₗ b.args)...)
conjed(a::AddArray) = +ₗ(conjed.(a.args)...)

matprod(x, y) = x * y + x * y
function mul_eltype(a::AbstractMatrix, b::AbstractVecOrMat)
    return Base.promote_op(matprod, eltype(a), eltype(b))
end
function mul_axes(a::AbstractMatrix, b::AbstractVecOrMat)
    return (axes(a, 1), axes(b, ndims(b)))
end

struct MulArray{T, N, A <: AbstractMatrix, B <: AbstractArray{<:Any, N}} <:
    AbstractLazyArray{T, N}
    a::A
    b::B
    function MulArray(a::AbstractMatrix, b::AbstractVecOrMat)
        T = mul_eltype(a, b)
        return new{T, ndims(b), typeof(a), typeof(b)}(a, b)
    end
end
const MulVector{T, A <: AbstractMatrix, B <: AbstractVector} = MulArray{T, 1, A, B}
const MulMatrix{T, A <: AbstractMatrix, B <: AbstractMatrix} = MulArray{T, 2, A, B}
const MulVecOrMat{T} = Union{MulVector{T}, MulMatrix{T}}
MulVector(a::AbstractMatrix, b::AbstractVector) = MulArray(a, b)
MulMatrix(a::AbstractMatrix, b::AbstractMatrix) = MulArray(a, b)

Base.axes(a::MulArray) = mul_axes(a.a, a.b)
Base.size(a::MulArray) = length.(axes(a))

Base.similar(a::MulArray) = similar(a, eltype(a))
Base.similar(a::MulArray, ax::Tuple) = similar(a, eltype(a), ax)
Base.similar(a::MulArray, elt::Type) = similar_mul(a, elt)
function Base.similar(
        a::MulArray,
        elt::Type,
        ax::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}},
    )
    return similar_mul(a, elt, ax)
end
Base.similar(a::MulArray, elt::Type, ax) = similar_mul(a, elt, ax)
# TODO: Make use of both arguments to determine the output, maybe
# using `LinearAlgebra.matprod_dest(a.a, a.b, elt)`?
similar_mul(a::MulArray, elt::Type) = similar(a.b, elt)
similar_mul(a::MulArray, elt::Type, ax) = similar(a.b, elt, ax)

TI.iscall(::MulArray) = true
TI.operation(::MulArray) = *
TI.arguments(a::MulArray) = (a.a, a.b)

function TA.add!(dest::AbstractArray, src::MulArray, α::Number, β::Number)
    LA.mul!(dest, src.a, src.b, α, β)
    return dest
end

conjed(a::MulArray) = *ₗ(conjed(a.a), conjed(a.b))
