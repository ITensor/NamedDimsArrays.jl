# TODO: Move this file to TensorAlgebra.jl.

import Base.Broadcast as BC
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

lazy_function(f) = error("No lazy function defined for `$f`.")
lazy_function(::typeof(+)) = +ₗ
lazy_function(::typeof(-)) = -ₗ
lazy_function(::typeof(*)) = *ₗ
lazy_function(::typeof(/)) = /ₗ
lazy_function(::typeof(\)) = \ₗ
lazy_function(::typeof(conj)) = conjed

broadcast_is_linear(f, args...) = false
broadcast_is_linear(::typeof(+), ::Base.AbstractArrayOrBroadcasted...) = true
broadcast_is_linear(::typeof(-), ::Base.AbstractArrayOrBroadcasted) = true
function broadcast_is_linear(
        ::typeof(-), ::Base.AbstractArrayOrBroadcasted, ::Base.AbstractArrayOrBroadcasted
    )
    return true
end
broadcast_is_linear(::typeof(*), ::Number, ::Base.AbstractArrayOrBroadcasted) = true
broadcast_is_linear(::typeof(\), ::Number, ::Base.AbstractArrayOrBroadcasted) = true
broadcast_is_linear(::typeof(*), ::Base.AbstractArrayOrBroadcasted, ::Number) = true
broadcast_is_linear(::typeof(/), ::Base.AbstractArrayOrBroadcasted, ::Number) = true
function broadcast_is_linear(
        ::typeof(*), ::Base.AbstractArrayOrBroadcasted, ::Base.AbstractArrayOrBroadcasted
    )
    return false
end
broadcast_is_linear(::typeof(*), ::Number, ::Number) = true
broadcast_is_linear(::typeof(conj), ::Base.AbstractArrayOrBroadcasted) = true
function is_linear(bc::BC.Broadcasted)
    return broadcast_is_linear(bc.f, bc.args...) && all(is_linear, bc.args)
end

to_linear(x) = x
to_linear(bc::BC.Broadcasted) = lazy_function(bc.f)(to_linear.(bc.args)...)
to_broadcasted(x) = x
function to_broadcasted(a::AbstractArray)
    !(BC.BroadcastStyle(typeof(a)) isa LazyArrayStyle) && return a
    return BC.broadcasted(TI.operation(a), TI.arguments(a)...)
end

# For lazy arrays, define Broadcast methods in terms of lazy operations.
struct LazyArrayStyle{N, Style <: BC.AbstractArrayStyle{N}} <: BC.AbstractArrayStyle{N}
    style::Style
end
function LazyArrayStyle{N, Style}(::Val{M}) where {Style <: BC.AbstractArrayStyle{N}} where {N, M}
    return LazyArrayStyle(Style(Val(M)))
end
function BC.BroadcastStyle(style1::LazyArrayStyle, style2::LazyArrayStyle)
    style = BC.BroadcastStyle(style1.style, style2.style)
    style ≡ BC.Unknown() && return BC.Unknown()
    return LazyArrayStyle(style)
end
function Base.similar(bc::BC.Broadcasted{<:LazyArrayStyle}, elt::Type, ax)
    return similar(BC.Broadcasted(bc.style.style, bc.f, bc.args, bc.axes), elt, ax)
end
# Backup definition, for broadcast operations that don't preserve LazyArrays
# (such as nonlinear operations), convert back to Broadcasted expressions.
function BC.broadcasted(::LazyArrayStyle, f, args...)
    return BC.broadcasted(f, to_broadcasted.(args)...)
end
function BC.broadcasted(
        ::LazyArrayStyle,
        ::typeof(+),
        a::AbstractArray,
        b::AbstractArray,
    )
    return a +ₗ b
end
function BC.broadcasted(
        ::LazyArrayStyle,
        ::typeof(+),
        a::AbstractArray,
        b::BC.Broadcasted,
    )
    is_linear(b) || return BC.broadcasted(+, to_broadcasted(a), b)
    return a +ₗ to_linear(b)
end
function BC.broadcasted(
        ::LazyArrayStyle,
        ::typeof(+),
        a::BC.Broadcasted,
        b::AbstractArray,
    )
    is_linear(a) || return BC.broadcasted(+, a, to_broadcasted(b))
    return to_linear(a) +ₗ b
end
function BC.broadcasted(
        ::LazyArrayStyle,
        ::typeof(+),
        a::BC.Broadcasted,
        b::BC.Broadcasted,
    )
    return error("Not implemented")
end
function BC.broadcasted(
        ::LazyArrayStyle, ::typeof(*), α::Number, a::AbstractArray
    )
    return α *ₗ a
end
function BC.broadcasted(
        ::LazyArrayStyle, ::typeof(*), a::AbstractArray, α::Number
    )
    return a *ₗ α
end
function BC.broadcasted(
        ::LazyArrayStyle, ::typeof(\), α::Number, a::AbstractArray
    )
    return α \ₗ a
end
function BC.broadcasted(
        ::LazyArrayStyle, ::typeof(/), a::AbstractArray, α::Number
    )
    return a /ₗ α
end
function BC.broadcasted(::LazyArrayStyle, ::typeof(-), a::AbstractArray)
    return -ₗ(a)
end
function BC.broadcasted(
        ::LazyArrayStyle, ::typeof(conj), a::AbstractArray
    )
    return conjed(a)
end

function show_lazy(io::IO, a::AbstractArray)
    print(io, TI.operation(a), "(", join(TI.arguments(a), ", "), ")")
    return nothing
end
function show_lazy(io::IO, mime::MIME"text/plain", a::AbstractArray)
    summary(io, a)
    println(io, ":")
    show(io, a)
    return nothing
end

struct ScaledArray{T, N, P <: AbstractArray{<:Any, N}, C <: Number} <: AbstractArray{T, N}
    coeff::C
    parent::P
    function ScaledArray(coeff::Number, a::AbstractArray)
        T = scaled_eltype(coeff, a)
        return new{T, ndims(a), typeof(a), typeof(coeff)}(coeff, a)
    end
end
scaled_eltype(coeff, a::AbstractArray) = Base.promote_op(*, typeof(coeff), eltype(a))
Base.axes(a::ScaledArray) = axes(a.parent)
Base.size(a::ScaledArray) = size(a.parent)

Base.similar(a::ScaledArray) = similar(a.parent)
Base.similar(a::ScaledArray, elt::Type) = similar(a.parent, elt)
Base.similar(a::ScaledArray, ax) = similar(a.parent, ax)
Base.similar(a::ScaledArray, elt::Type, ax) = similar_scaled(a, elt, ax)
Base.similar(a::ScaledArray, elt::Type, ax::Dims) = similar_scaled(a, elt, ax)
similar_scaled(a::ScaledArray, elt::Type, ax) = similar(a.parent, elt, ax)

function Base.copyto!(dest::AbstractArray, src::ScaledArray)
    TA.add!(dest, src, true, false)
    return dest
end
function TA.add!(dest::AbstractArray, src::ScaledArray, α::Number, β::Number)
    TA.add!(dest, src.parent, src.coeff * α, β)
    return dest
end
BC.materialize(a::ScaledArray) = copy(a)
TI.iscall(::ScaledArray) = true
TI.operation(::ScaledArray) = *
TI.arguments(a::ScaledArray) = (a.coeff, a.parent)

*ₗ(α::Number, a::ScaledArray) = (α * a.coeff) *ₗ a.parent
*ₗ(a::ScaledArray, b::ScaledArray) = (a.coeff * b.coeff) *ₗ (a.parent *ₗ b.parent)
*ₗ(a::AbstractArray, b::ScaledArray) = b.coeff *ₗ (a *ₗ b.parent)
*ₗ(a::ScaledArray, b::AbstractArray) = a.coeff *ₗ (a.parent *ₗ b)
conjed(a::ScaledArray) = conj(a.coeff) *ₗ conjed(a.parent)

function BC.BroadcastStyle(arrayt::Type{<:ScaledArray{<:Any, <:Any, P}}) where {P}
    return LazyArrayStyle(BC.BroadcastStyle(P))
end

Base.show(io::IO, a::ScaledArray) = show_lazy(io, a)
Base.show(io::IO, mime::MIME"text/plain", a::ScaledArray) = show_lazy(io, mime, a)

struct ConjArray{T, N, P <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::P
end
Base.axes(a::ConjArray) = axes(a.parent)
Base.size(a::ConjArray) = size(a.parent)

Base.similar(a::ConjArray, elt::Type) = similar(a.parent, elt)
Base.similar(a::ConjArray, elt::Type, ax) = similar_conj(a, elt, ax)
Base.similar(a::ConjArray, elt::Type, ax::Dims) = similar_conj(a, elt, ax)
similar_conj(a::ConjArray, elt::Type, ax) = similar(a.parent, elt, ax)

function Base.copyto!(dest::AbstractArray, src::ConjArray)
    TA.add!(dest, src, true, false)
    return dest
end
BC.materialize(a::ConjArray) = copy(a)

using StridedViews: StridedViews, StridedView, isstrided
StridedViews.isstrided(a::ConjArray) = isstrided(a.parent)
StridedViews.StridedView(a::ConjArray) = conj(StridedView(a.parent))

TI.iscall(::ConjArray) = true
TI.operation(::ConjArray) = conj
TI.arguments(a::ConjArray) = (a.parent,)

conjed(a::ConjArray) = a.parent

function BC.BroadcastStyle(arrayt::Type{<:ConjArray{<:Any, <:Any, P}}) where {P}
    return LazyArrayStyle(BC.BroadcastStyle(P))
end

Base.show(io::IO, a::ConjArray) = show_lazy(io, a)
Base.show(io::IO, mime::MIME"text/plain", a::ConjArray) = show_lazy(io, mime, a)

struct AddArray{T, N, Args <: Tuple{Vararg{AbstractArray{<:Any, N}}}} <: AbstractArray{T, N}
    args::Args
    function AddArray(args::AbstractArray{<:Any, N}...) where {N}
        T = add_eltype(args...)
        return new{T, N, typeof(args)}(args)
    end
end

function add_eltype(args::AbstractArray{<:Any, N}...) where {N}
    return Base.promote_op(+, eltype.(args)...)
end
using Base.Broadcast: broadcasted
add_axes(args::AbstractArray{<:Any, N}...) where {N} = BC.combine_axes(args...)
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
Base.similar(a::AddArray, elt::Type, ax::Dims) = similar_add(a, elt, ax)
Base.similar(a::AddArray, elt::Type, ax) = similar_add(a, elt, ax)
using Base.Broadcast: broadcasted
similar_add(a::AddArray, elt) = similar(BC.Broadcasted(+, a.args), elt)
function similar_add(a::AddArray, elt::Type, ax)
    return similar(BC.Broadcasted(+, a.args), elt, ax)
end

function Base.copyto!(dest::AbstractArray, src::AddArray)
    TA.add!(dest, src, true, false)
    return dest
end
function TA.add!(dest::AbstractArray, src::AddArray, α::Number, β::Number)
    TA.add!(dest, first(src.args), α, β)
    for a in Base.tail(src.args)
        TA.add!(dest, a, α, true)
    end
    return dest
end
BC.materialize(a::AddArray) = copy(a)
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

function BC.BroadcastStyle(arrayt::Type{<:AddArray{<:Any, <:Any, Args}}) where {Args}
    style = Base.promote_op(BC.combine_styles, fieldtypes(Args)...)()
    return LazyArrayStyle(style)
end

Base.show(io::IO, a::AddArray) = show_lazy(io, a)
Base.show(io::IO, mime::MIME"text/plain", a::AddArray) = show_lazy(io, mime, a)

struct MulArray{T, N, A <: AbstractArray, B <: AbstractArray} <: AbstractArray{T, N}
    a::A
    b::B
    function MulArray(a::AbstractArray, b::AbstractArray)
        T = mul_eltype(a, b)
        N = mul_ndims(a, b)
        return new{T, N, typeof(a), typeof(b)}(a, b)
    end
end

# Same as `LinearAlgebra.matprod`, but duplicated here since it is private.
matprod(x, y) = x * y + x * y
function mul_eltype(a::AbstractArray, b::AbstractArray)
    return Base.promote_op(matprod, eltype(a), eltype(b))
end
mul_ndims(a::AbstractArray, b::AbstractArray) = ndims(b)
function mul_axes(a::AbstractArray, b::AbstractArray)
    return (axes(a, 1), axes(b, ndims(b)))
end
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
Base.similar(a::MulArray, elt::Type, ax::Dims) = similar_mul(a, elt, ax)
# TODO: Make use of both arguments to determine the output, maybe
# using `LinearAlgebra.matprod_dest(a.a, a.b, elt)`?
similar_mul(a::MulArray, elt::Type) = similar(a.b, elt)
similar_mul(a::MulArray, elt::Type, ax) = similar(a.b, elt, ax)
BC.materialize(a::MulArray) = copy(a)

TI.iscall(::MulArray) = true
TI.operation(::MulArray) = *
TI.arguments(a::MulArray) = (a.a, a.b)

function Base.copyto!(dest::AbstractArray, src::MulArray)
    TA.add!(dest, src, true, false)
    return dest
end
function TA.add!(dest::AbstractArray, src::MulArray, α::Number, β::Number)
    # We materialize the arguments here to avoid nested lazy evaluation.
    # Rewrite rules should make it so that `MulArray` is a "leaf` node of the
    # expression tree.
    LA.mul!(dest, BC.materialize.((src.a, src.b))..., α, β)
    return dest
end

conjed(a::MulArray) = *ₗ(conjed(a.a), conjed(a.b))

function BC.BroadcastStyle(arrayt::Type{<:MulArray{<:Any, <:Any, A, B}}) where {A, B}
    style = Base.promote_op(BC.combine_styles, A, B)()
    return LazyArrayStyle(style)
end
to_broadcasted(a::MulArray) = a.a * a.b

Base.show(io::IO, a::MulArray) = show_lazy(io, a)
Base.show(io::IO, mime::MIME"text/plain", a::MulArray) = show_lazy(io, mime, a)
