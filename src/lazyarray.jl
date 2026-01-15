# TODO: Move this file to TensorAlgebra.jl.

import Base.Broadcast as BC
import LinearAlgebra as LA
import TensorAlgebra as TA
import TermInterface as TI

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
conjed(a::AbstractArray{<:Real}) = a

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

# Base overloads for lazy arrays.
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

# Generic constructors, accessors, and properties for ScaledArrays.
*ₗ(α::Number, a::AbstractArray) = ScaledArray(α, a)
unscaled(a::AbstractArray) = a
unscaled_type(arrayt::Type{<:AbstractArray}) = Base.promote_op(unscaled, arrayt)
coeff(a::AbstractArray) = true
coeff_type(arrayt::Type{<:AbstractArray}) = Base.promote_op(coeff, arrayt)
scaled_eltype(coeff::Number, a::AbstractArray) =
    Base.promote_op(*, typeof(coeff), eltype(a))

# Base overloads for ScaledArrays.
axes_scaled(a::AbstractArray) = axes(unscaled(a))
size_scaled(a::AbstractArray) = size(unscaled(a))
similar_scaled(a::AbstractArray) = similar(unscaled(a))
similar_scaled(a::AbstractArray, elt::Type) = similar(unscaled(a), elt)
similar_scaled(a::AbstractArray, ax) = similar(unscaled(a), ax)
similar_scaled(a::AbstractArray, elt::Type, ax) = similar(unscaled(a), elt, ax)
function copyto!_scaled(dest::AbstractArray, src::AbstractArray)
    TA.add!(dest, src, true, false)
    return dest
end
show_scaled(io::IO, a::AbstractArray) = show_lazy(io, a)
show_scaled(io::IO, mime::MIME"text/plain", a::AbstractArray) = show_lazy(io, mime, a)

# Base.Broadcast overloads for ScaledArrays.
materialize_scaled(a::AbstractArray) = copy(a)
function BroadcastStyle_scaled(arrayt::Type{<:AbstractArray})
    return LazyArrayStyle(BC.BroadcastStyle(unscaled_type(arrayt)))
end

# LinearAlgebra overloads for ScaledArrays.
mul!_scaled(dest::AbstractArray, a::AbstractArray, b::AbstractArray, α::Number, β::Number) =
    LA.mul!(dest, unscaled(a), unscaled(b), coeff(a) * coeff(b) * α, β)

# Lazy operations for ScaledArrays.
mulled_scaled(α::Number, a::AbstractArray) = (α * coeff(a)) *ₗ unscaled(a)
mulled_scaled(a::AbstractArray, b::AbstractArray) =
    (coeff(a) * coeff(b)) *ₗ (unscaled(a) *ₗ unscaled(b))
conjed_scaled(a::AbstractArray) = conj(coeff(a)) *ₗ conjed(unscaled(a))

# TensorAlgebra overloads for ScaledArrays.
function add!_scaled(dest::AbstractArray, src::AbstractArray, α::Number, β::Number)
    TA.add!(dest, unscaled(src), coeff(src) * α, β)
    return dest
end

# TermInterface overloads for ScaledArrays.
iscall_scaled(::AbstractArray) = true
operation_scaled(::AbstractArray) = *
arguments_scaled(a::AbstractArray) = (coeff(a), unscaled(a))

macro scaledarray_type(ScaledArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            struct $ScaledArray{T, N, P <: AbstractArray{<:Any, N}, C <: Number} <:
                $AbstractArray{T, N}
                coeff::C
                parent::P
                function $ScaledArray(coeff::Number, a::AbstractArray)
                    T = scaled_eltype(coeff, a)
                    return new{T, ndims(a), typeof(a), typeof(coeff)}(coeff, a)
                end
            end
            NamedDimsArrays.unscaled(a::$ScaledArray) = a.parent
            NamedDimsArrays.unscaled_type(arrayt::Type{<:$ScaledArray}) =
                fieldtype(arrayt, :parent)
            NamedDimsArrays.coeff(a::$ScaledArray) = a.coeff
            NamedDimsArrays.coeff_type(arrayt::Type{<:$ScaledArray}) = fieldtype(arrayt, :coeff)
        end
    )
end

macro scaledarray_base(ScaledArray)
    return esc(
        quote
            Base.axes(a::$ScaledArray) = NamedDimsArrays.axes_scaled(a)
            Base.size(a::$ScaledArray) = NamedDimsArrays.size_scaled(a)
            Base.similar(a::$ScaledArray) = NamedDimsArrays.similar_scaled(a)
            Base.similar(a::$ScaledArray, elt::Type) = NamedDimsArrays.similar_scaled(a, elt)
            Base.similar(a::$ScaledArray, ax) = NamedDimsArrays.similar_scaled(a, ax)
            Base.similar(a::$ScaledArray, elt::Type, ax) = NamedDimsArrays.similar_scaled(a, elt, ax)
            Base.similar(a::$ScaledArray, elt::Type, ax::Dims) = NamedDimsArrays.similar_scaled(a, elt, ax)
            Base.copyto!(dest::AbstractArray, src::$ScaledArray) = NamedDimsArrays.copyto!_scaled(dest, src)
            Base.show(io::IO, a::$ScaledArray) = NamedDimsArrays.show_scaled(io, a)
            Base.show(io::IO, mime::MIME"text/plain", a::$ScaledArray) = NamedDimsArrays.show_scaled(io, mime, a)
        end
    )
end

macro scaledarray_broadcast(ScaledArray)
    return esc(
        quote
            Base.Broadcast.materialize(a::$ScaledArray) = NamedDimsArrays.materialize_scaled(a)
            Base.Broadcast.BroadcastStyle(arrayt::Type{<:$ScaledArray}) =
                NamedDimsArrays.BroadcastStyle_scaled(arrayt)
        end
    )
end

macro scaledarray_linearalgebra(ScaledArray)
    return esc(
        quote
            function LinearAlgebra.mul!(
                    dest::AbstractMatrix, a::$ScaledArray{<:Any, 2}, b::$ScaledArray{<:Any, 2},
                    α::Number, β::Number,
                )
                return NamedDimsArrays.mul!_scaled(dest, a, b, α, β)
            end
            function LinearAlgebra.mul!(
                    dest::AbstractMatrix, a::AbstractMatrix, b::$ScaledArray{<:Any, 2},
                    α::Number, β::Number,
                )
                return NamedDimsArrays.mul!_scaled(dest, a, b, α, β)
            end
            function LinearAlgebra.mul!(
                    dest::AbstractMatrix, a::$ScaledArray{<:Any, 2}, b::AbstractMatrix,
                    α::Number, β::Number,
                )
                return NamedDimsArrays.mul!_scaled(dest, a, b, α, β)
            end
        end
    )
end

macro scaledarray_lazy(ScaledArray)
    return esc(
        quote
            NamedDimsArrays.:(*ₗ)(α::Number, a::$ScaledArray) =
                NamedDimsArrays.mulled_scaled(α, a)
            NamedDimsArrays.:(*ₗ)(a::$ScaledArray, b::$ScaledArray) =
                NamedDimsArrays.mulled_scaled(a, b)
            NamedDimsArrays.:(*ₗ)(a::AbstractArray, b::$ScaledArray) =
                NamedDimsArrays.mulled_scaled(a, b)
            NamedDimsArrays.:(*ₗ)(a::$ScaledArray, b::AbstractArray) =
                NamedDimsArrays.mulled_scaled(a, b)
            NamedDimsArrays.conjed(a::$ScaledArray) = NamedDimsArrays.conjed_scaled(a)
        end
    )
end

macro scaledarray_tensoralgebra(ScaledArray)
    return esc(
        quote
            TensorAlgebra.add!(dest::AbstractArray, src::$ScaledArray, α::Number, β::Number) =
                NamedDimsArrays.add!_scaled(dest, src, α, β)
        end
    )
end

macro scaledarray_terminterface(ScaledArray)
    return esc(
        quote
            TermInterface.iscall(a::$ScaledArray) = NamedDimsArrays.iscall_scaled(a)
            TermInterface.operation(a::$ScaledArray) = NamedDimsArrays.operation_scaled(a)
            TermInterface.arguments(a::$ScaledArray) = NamedDimsArrays.arguments_scaled(a)
        end
    )
end

macro scaledarray(ScaledArray)
    return esc(
        quote
            NamedDimsArrays.@scaledarray_base $ScaledArray
            NamedDimsArrays.@scaledarray_broadcast $ScaledArray
            NamedDimsArrays.@scaledarray_lazy $ScaledArray
            NamedDimsArrays.@scaledarray_linearalgebra $ScaledArray
            NamedDimsArrays.@scaledarray_tensoralgebra $ScaledArray
            NamedDimsArrays.@scaledarray_terminterface $ScaledArray
        end
    )
end

# Generic constructors for ConjArrays.
conjed(a::AbstractArray) = ConjArray(a)
conjed_type(arrayt::Type{<:AbstractArray}) = Base.promote_op(conj, arrayt)

# Base overloads for ConjArrays.
axes_conj(a::AbstractArray) = axes(conjed(a))
size_conj(a::AbstractArray) = size(conjed(a))
similar_conj(a::AbstractArray, elt::Type) = similar(conjed(a), elt)
similar_conj(a::AbstractArray, elt::Type, ax) = similar(conjed(a), elt, ax)
similar_conj(a::AbstractArray, ax) = similar(conjed(a), ax)
function copyto!_conj(dest::AbstractArray, src::AbstractArray)
    TA.add!(dest, src, true, false)
    return dest
end
show_conj(io::IO, a::AbstractArray) = show_lazy(io, a)
show_conj(io::IO, mime::MIME"text/plain", a::AbstractArray) = show_lazy(io, mime, a)

# Base.Broadcast overloads for ConjArrays.
materialize_conj(a::AbstractArray) = copy(a)
function BroadcastStyle_conj(arrayt::Type{<:AbstractArray})
    return LazyArrayStyle(BC.BroadcastStyle(conjed_type(arrayt)))
end

# StridedViews overloads for ConjArrays.
using StridedViews: StridedView, isstrided
isstrided_conj(a::AbstractArray) = isstrided(conjed(a))
StridedView_conj(a::AbstractArray) = conj(StridedView(conjed(a)))

# TermInterface overloads for ConjArrays.
iscall_conj(::AbstractArray) = true
operation_conj(::AbstractArray) = conj
arguments_conj(a::AbstractArray) = (conjed(a),)

macro conjarray_type(ConjArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            struct $ConjArray{T, N, P <: AbstractArray{T, N}} <: $AbstractArray{T, N}
                parent::P
            end
            conjed(a::$ConjArray) = a.parent
        end
    )
end

macro conjarray_base(ConjArray)
    return esc(
        quote
            Base.axes(a::$ConjArray) = axes_conj(a)
            Base.size(a::$ConjArray) = size_conj(a)
            Base.similar(a::$ConjArray, elt::Type) = similar_conj(a, elt)
            Base.similar(a::$ConjArray, elt::Type, ax) = similar_conj(a, elt, ax)
            Base.similar(a::$ConjArray, elt::Type, ax::Dims) = similar_conj(a, elt, ax)
            Base.copyto!(dest::AbstractArray, src::$ConjArray) = copyto!_conj(dest, src)
            Base.show(io::IO, a::$ConjArray) = show_conj(io, a)
            Base.show(io::IO, mime::MIME"text/plain", a::$ConjArray) = show_conj(io, mime, a)
        end
    )
end

macro conjarray_broadcast(ConjArray)
    return esc(
        quote
            Base.Broadcast.materialize(a::$ConjArray) = materialize_conj(a)
            Base.Broadcast.BroadcastStyle(arrayt::Type{<:$ConjArray}) =
                BroadcastStyle_conj(arrayt)
        end
    )
end

macro conjarray_stridedviews(ConjArray)
    return esc(
        quote
            StridedViews.isstrided(a::$ConjArray) = isstrided_conj(a)
            StridedViews.StridedView(a::$ConjArray) = StridedView_conj(a)
        end
    )
end

macro conjarray_terminterface(ConjArray)
    return esc(
        quote
            TI.iscall(::$ConjArray) = true
            TI.operation(::$ConjArray) = conj
            TI.arguments(a::$ConjArray) = (a.parent,)
        end
    )
end

macro conjarray(ConjArray)
    return esc(
        quote
            NamedDimsArrays.@conjarray_base $ConjArray
            NamedDimsArrays.@conjarray_broadcast $ConjArray
            NamedDimsArrays.@conjarray_stridedviews $ConjArray
            NamedDimsArrays.@conjarray_terminterface $ConjArray
        end
    )
end

# Generic constructors, accessors, and properties for AddArrays.
+ₗ(a::AbstractArray, b::AbstractArray) = AddArray(a, b)
addends(a::AbstractArray) = (a,)
addends_type(arrayt::Type{<:AbstractArray}) = Tuple{arrayt}
add_eltype(args::AbstractArray{<:Any, N}...) where {N} = Base.promote_op(+, eltype.(args)...)

# Base overloads for AddArrays.
add_axes(args::AbstractArray{<:Any, N}...) where {N} = BC.combine_axes(args...)
axes_add(a::AbstractArray) = add_axes(addends(a)...)
size_add(a::AbstractArray) = length.(axes_add(a))
similar_add(a::AbstractArray) = similar(a, eltype(a))
similar_add(a::AbstractArray, ax::Tuple) = similar(a, eltype(a), ax)
similar_add(a::AbstractArray, elt::Type) = similar(BC.Broadcasted(+, addends(a)), elt)
similar_add(a::AbstractArray, elt::Type, ax) = similar(BC.Broadcasted(+, addends(a)), elt, ax)
function copyto!_add(dest::AbstractArray, src::AbstractArray)
    TA.add!(dest, src, true, false)
    return dest
end
show_add(io::IO, a::AbstractArray) = show_lazy(io, a)
show_add(io::IO, mime::MIME"text/plain", a::AbstractArray) = show_lazy(io, mime, a)

# Base.Broadcast overloads for AddArrays.
materialize_add(a::AbstractArray) = copy(a)
function BroadcastStyle_add(arrayt::Type{<:AbstractArray})
    args_type = addends_type(arrayt)
    style = Base.promote_op(BC.combine_styles, fieldtypes(args_type)...)()
    return LazyArrayStyle(style)
end

# TensorAlgebra overloads for AddArrays.
function add!_add(dest::AbstractArray, src::AbstractArray, α::Number, β::Number)
    args = addends(src)
    TA.add!(dest, first(args), α, β)
    for a in Base.tail(args)
        TA.add!(dest, a, α, true)
    end
    return dest
end

# Lazy operations for AddArrays.
added_add(a::AbstractArray, b::AbstractArray) = AddArray((addends(a)..., addends(b)...)...)
mulled_add(α::Number, a::AbstractArray) = +ₗ((α .*ₗ addends(a))...)
## mulled_add(a::AbstractArray, b::AbstractArray) = +ₗ((Ref(a) .*ₗ addends(b))...)
## mulled_add(a::AddArray, b::AbstractArray) = +ₗ((addends(a) .*ₗ Ref(b))...)
## mulled_add(a::AddArray, b::AddArray) = +ₗ((Ref(a) .*ₗ addends(b))...)
conjed_add(a::AbstractArray) = +ₗ(conjed.(addends(a))...)

# TermInterface overloads for AddArrays.
iscall_add(::AbstractArray) = true
operation_add(::AbstractArray) = +
arguments_add(a::AbstractArray) = addends(a)

macro addarray_type(AddArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            struct $AddArray{T, N, Args <: Tuple{Vararg{AbstractArray{<:Any, N}}}} <:
                $AbstractArray{T, N}
                args::Args
                function $AddArray(args::AbstractArray{<:Any, N}...) where {N}
                    T = NamedDimsArrays.add_eltype(args...)
                    return new{T, N, typeof(args)}(args)
                end
            end
            NamedDimsArrays.addends(a::$AddArray) = a.args
            NamedDimsArrays.addends_type(arrayt::Type{<:$AddArray}) = fieldtype(arrayt, :args)
        end
    )
end

macro addarray_base(AddArray)
    return esc(
        quote
            Base.axes(a::$AddArray) = NamedDimsArrays.axes_add(a)
            Base.size(a::$AddArray) = NamedDimsArrays.size_add(a)
            Base.similar(a::$AddArray) = NamedDimsArrays.similar_add(a)
            Base.similar(a::$AddArray, ax::Tuple) = NamedDimsArrays.similar_add(a, ax)
            Base.similar(a::$AddArray, elt::Type) = NamedDimsArrays.similar_add(a, elt)
            function Base.similar(
                    a::$AddArray, elt::Type,
                    ax::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}},
                )
                return NamedDimsArrays.similar_add(a, elt, ax)
            end
            Base.similar(a::$AddArray, elt::Type, ax::Dims) =
                NamedDimsArrays.similar_add(a, elt, ax)
            Base.similar(a::$AddArray, elt::Type, ax) =
                NamedDimsArrays.similar_add(a, elt, ax)
            Base.copyto!(dest::AbstractArray, src::$AddArray) =
                NamedDimsArrays.copyto!_add(dest, src)
            Base.show(io::IO, a::$AddArray) =
                NamedDimsArrays.show_add(io, a)
            Base.show(io::IO, mime::MIME"text/plain", a::$AddArray) =
                NamedDimsArrays.show_add(io, mime, a)
        end
    )
end

macro addarray_broadcast(AddArray)
    return esc(
        quote
            Base.Broadcast.materialize(a::$AddArray) = NamedDimsArrays.materialize_add(a)
            Base.Broadcast.BroadcastStyle(arrayt::Type{<:$AddArray}) =
                NamedDimsArrays.BroadcastStyle_add(arrayt)
        end
    )
end

macro addarray_lazy(AddArray)
    return esc(
        quote
            NamedDimsArrays.:(+ₗ)(a::AbstractArray, b::$AddArray) = NamedDimsArrays.added_add(a, b)
            NamedDimsArrays.:(+ₗ)(a::$AddArray, b::AbstractArray) = NamedDimsArrays.added_add(a, b)
            NamedDimsArrays.:(+ₗ)(a::$AddArray, b::$AddArray) = NamedDimsArrays.added_add(a, b)
            NamedDimsArrays.:(*ₗ)(α::Number, a::$AddArray) = NamedDimsArrays.mulled_add(α, a)
            NamedDimsArrays.:(*ₗ)(a::AbstractArray, b::$AddArray) = NamedDimsArrays.mulled_add(a, b)
            NamedDimsArrays.:(*ₗ)(a::$AddArray, b::AbstractArray) = NamedDimsArrays.mulled_add(a, b)
            NamedDimsArrays.:(*ₗ)(a::$AddArray, b::$AddArray) = NamedDimsArrays.mulled_add(a, b)
            NamedDimsArrays.conjed(a::$AddArray) = NamedDimsArrays.conjed_add(a)
        end
    )
end

macro addarray_tensoralgebra(AddArray)
    return esc(
        quote
            TensorAlgebra.add!(dest::AbstractArray, src::$AddArray, α::Number, β::Number) =
                NamedDimsArrays.add!_add(dest, src, α, β)
        end
    )
end

macro addarray_terminterface(AddArray)
    return esc(
        quote
            TermInterface.iscall(a::$AddArray) = NamedDimsArrays.iscall_add(a)
            TermInterface.operation(a::$AddArray) = NamedDimsArrays.operation_add(a)
            TermInterface.arguments(a::$AddArray) = NamedDimsArrays.arguments_add(a)
        end
    )
end

macro addarray(AddArray)
    return esc(
        quote
            NamedDimsArrays.@addarray_base $AddArray
            NamedDimsArrays.@addarray_broadcast $AddArray
            NamedDimsArrays.@addarray_lazy $AddArray
            NamedDimsArrays.@addarray_tensoralgebra $AddArray
            NamedDimsArrays.@addarray_terminterface $AddArray
        end
    )
end

# Generic constructors, accessors, and properties for MulArrays.
*ₗ(a::AbstractArray, b::AbstractArray) = MulArray(a, b)
mulled_left(a::AbstractArray) = error("No mulled_left defined for type $(typeof(a))")
mulled_right(a::AbstractArray) = error("No mulled_right defined for type $(typeof(a))")
mulled_left_type(arrayt::Type{<:AbstractArray}) = Base.promote_op(mulled_left, arrayt)
mulled_right_type(arrayt::Type{<:AbstractArray}) = Base.promote_op(mulled_right, arrayt)
# Same as `LinearAlgebra.matprod`, but duplicated here since it is private.
matprod(x, y) = x * y + x * y
mul_eltype(a::AbstractArray, b::AbstractArray) = Base.promote_op(matprod, eltype(a), eltype(b))
mul_ndims(a::AbstractArray, b::AbstractArray) = ndims(b)
mul_axes(a::AbstractArray, b::AbstractArray) = (axes(a, 1), axes(b, ndims(b)))

# Base overloads for MulArrays.
axes_mul(a::AbstractArray) = mul_axes(mulled_left(a), mulled_right(a))
size_mul(a::AbstractArray) = length.(axes_mul(a))
similar_mul(a::AbstractArray) = similar(a, eltype(a))
similar_mul(a::AbstractArray, ax::Tuple) = similar(a, eltype(a), ax)
# TODO: Make use of both arguments to determine the output, maybe
# using `LinearAlgebra.matprod_dest(mulled_left(a), mulled_right(a), elt)`?
similar_mul(a::AbstractArray, elt::Type) = similar(mulled_right(a), elt)
similar_mul(a::AbstractArray, elt::Type, ax) = similar(mulled_right(a), elt, ax)
function copyto!_mul(dest::AbstractArray, src::AbstractArray)
    TA.add!(dest, src, true, false)
    return dest
end
show_mul(io::IO, a::AbstractArray) = show_lazy(io, a)
show_mul(io::IO, mime::MIME"text/plain", a::AbstractArray) = show_lazy(io, mime, a)

# Base.Broadcast overloads for MulArrays.
materialize_mul(a::AbstractArray) = copy(a)
function BroadcastStyle_mul(arrayt::Type{<:AbstractArray})
    A = mulled_left_type(arrayt)
    B = mulled_right_type(arrayt)
    style = Base.promote_op(BC.combine_styles, A, B)()
    return LazyArrayStyle(style)
end
to_broadcasted_mul(a::AbstractArray) = mulled_left(a) * mulled_right(a)

# TensorAlgebra overloads for MulArrays.
function add!_mul(dest::AbstractArray, src::AbstractArray, α::Number, β::Number)
    # We materialize the arguments here to avoid nested lazy evaluation.
    # Rewrite rules should make it so that `MulArray` is a "leaf` node of the
    # expression tree.
    LA.mul!(dest, BC.materialize.((mulled_left(src), mulled_right(src)))..., α, β)
    return dest
end

# Lazy operations for MulArrays.
conjed_mul(a::AbstractArray) = *ₗ(conjed(mulled_left(a)), conjed(mulled_right(a)))

# TermInterface overloads for MulArrays.
iscall_mul(::AbstractArray) = true
operation_mul(::AbstractArray) = *
arguments_mul(a::AbstractArray) = (mulled_left(a), mulled_right(a))

macro mularray_type(MulArray, AbstractArray = :AbstractArray)
    return esc(
        quote
            struct $MulArray{T, N, A <: AbstractArray, B <: AbstractArray} <:
                $AbstractArray{T, N}
                a::A
                b::B
                function $MulArray(a::AbstractArray, b::AbstractArray)
                    T = NamedDimsArrays.mul_eltype(a, b)
                    N = NamedDimsArrays.mul_ndims(a, b)
                    return new{T, N, typeof(a), typeof(b)}(a, b)
                end
            end
            NamedDimsArrays.mulled_left(a::$MulArray) = a.a
            NamedDimsArrays.mulled_right(a::$MulArray) = a.b
            NamedDimsArrays.mulled_left_type(arrayt::Type{<:$MulArray}) = fieldtype(arrayt, :a)
            NamedDimsArrays.mulled_right_type(arrayt::Type{<:$MulArray}) = fieldtype(arrayt, :b)
        end
    )
end

macro mularray_base(MulArray)
    return esc(
        quote
            Base.axes(a::$MulArray) = NamedDimsArrays.axes_mul(a)
            Base.size(a::$MulArray) = NamedDimsArrays.size_mul(a)
            Base.similar(a::$MulArray) = NamedDimsArrays.similar_mul(a)
            Base.similar(a::$MulArray, ax::Tuple) = NamedDimsArrays.similar_mul(a, ax)
            Base.similar(a::$MulArray, elt::Type) = NamedDimsArrays.similar_mul(a, elt)
            Base.similar(
                a::$MulArray, elt::Type,
                ax::Tuple{Union{Integer, Base.OneTo}, Vararg{Union{Integer, Base.OneTo}}},
            ) = NamedDimsArrays.similar_mul(a, elt, ax)
            Base.similar(a::$MulArray, elt::Type, ax) = NamedDimsArrays.similar_mul(a, elt, ax)
            Base.similar(a::$MulArray, elt::Type, ax::Dims) = NamedDimsArrays.similar_mul(a, elt, ax)
            Base.copyto!(dest::AbstractArray, src::$MulArray) = NamedDimsArrays.copyto!_mul(dest, src)
            Base.show(io::IO, a::$MulArray) = NamedDimsArrays.show_mul(io, a)
            Base.show(io::IO, mime::MIME"text/plain", a::$MulArray) = NamedDimsArrays.show_mul(io, mime, a)
        end
    )
end

macro mularray_broadcast(MulArray)
    return esc(
        quote
            Base.Broadcast.materialize(a::$MulArray) = NamedDimsArrays.materialize_mul(a)
            Base.Broadcast.BroadcastStyle(arrayt::Type{<:$MulArray}) =
                NamedDimsArrays.BroadcastStyle_mul(arrayt)
            NamedDimsArrays.to_broadcasted(a::$MulArray) = NamedDimsArrays.to_broadcasted_mul(a)
        end
    )
end

macro mularray_lazy(MulArray)
    return esc(
        quote
            NamedDimsArrays.conjed(a::$MulArray) = NamedDimsArrays.conjed_mul(a)
        end
    )
end

macro mularray_tensoralgebra(MulArray)
    return esc(
        quote
            TensorAlgebra.add!(dest::AbstractArray, src::$MulArray, α::Number, β::Number) =
                NamedDimsArrays.add!_mul(dest, src, α, β)
        end
    )
end

macro mularray_terminterface(MulArray)
    return esc(
        quote
            TermInterface.iscall(a::$MulArray) = NamedDimsArrays.iscall_mul(a)
            TermInterface.operation(a::$MulArray) = NamedDimsArrays.operation_mul(a)
            TermInterface.arguments(a::$MulArray) = NamedDimsArrays.arguments_mul(a)
        end
    )
end

macro mularray(MulArray)
    return esc(
        quote
            NamedDimsArrays.@mularray_base $MulArray
            NamedDimsArrays.@mularray_broadcast $MulArray
            NamedDimsArrays.@mularray_lazy $MulArray
            NamedDimsArrays.@mularray_tensoralgebra $MulArray
            NamedDimsArrays.@mularray_terminterface $MulArray
        end
    )
end

# Define types.
import LinearAlgebra, StridedViews, TensorAlgebra, TermInterface
@scaledarray_type ScaledArray
@scaledarray ScaledArray
@conjarray_type ConjArray
@conjarray ConjArray
@addarray_type AddArray
@addarray AddArray
@mularray_type MulArray
@mularray MulArray
