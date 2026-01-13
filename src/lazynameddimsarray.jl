using TermInterface: TermInterface as TI, operation, arguments

# Primitive constructors for lazy named dims array linear algebra operations.
+ₗ(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) = AddNamedDimsArray(a, b)
*ₗ(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) = MulNamedDimsArray(a, b)
*ₗ(α::Number, a::AbstractNamedDimsArray) = ScaledNamedDimsArray(α, a)
conjed(a::AbstractNamedDimsArray) = ConjNamedDimsArray(a)
conjed(a::AbstractNamedDimsArray{<:Real}) = a

abstract type AbstractLazyNamedDimsArray{T, N} <: AbstractNamedDimsArray{T, N} end

# For lazy named dims arrays, define Base methods in terms of lazy operations.
Base.:(+)(a::AbstractLazyNamedDimsArray, b::AbstractLazyNamedDimsArray) = a +ₗ b
Base.:(+)(a::AbstractLazyNamedDimsArray, b::AbstractNamedDimsArray) = a +ₗ b
Base.:(+)(a::AbstractNamedDimsArray, b::AbstractLazyNamedDimsArray) = a +ₗ b
Base.:(*)(α::Number, a::AbstractLazyNamedDimsArray) = α *ₗ a
Base.:(*)(a::AbstractLazyNamedDimsArray, α::Number) = a *ₗ α
Base.:(\)(α::Number, a::AbstractLazyNamedDimsArray) = α \ₗ a
Base.:(/)(a::AbstractLazyNamedDimsArray, α::Number) = a /ₗ α
Base.:(-)(a::AbstractLazyNamedDimsArray) = -ₗ(a)
Base.conj(a::AbstractLazyNamedDimsArray) = conjed(a)

function Base.similar(a::AbstractLazyNamedDimsArray, elt::Type)
    return similar(a, elt, axes(a))
end
function Base.similar(a::AbstractLazyNamedDimsArray, elt::Type, ax)
    return error("Not implemented.")
end

# Interface for materializing a lazy named dims array.
Base.copy(a::AbstractLazyNamedDimsArray) = copyto!(similar(a), a)
function Base.copyto!(dest::AbstractNamedDimsArray, src::AbstractLazyNamedDimsArray)
    return error("Not implemented.")
end

function Base.show(io::IO, a::AbstractLazyNamedDimsArray)
    print(io, operation(a), "(", join(arguments(a), ", "), ")")
    return nothing
end
function Base.show(io::IO, mime::MIME"text/plain", a::AbstractLazyNamedDimsArray)
    summary(io, a)
    println(io, ":")
    show(io, a)
    return nothing
end

struct ScaledNamedDimsArray{T, N, C <: Number, TP, P <: AbstractNamedDimsArray{TP, N}} <:
    AbstractLazyNamedDimsArray{T, N}
    coeff::C
    parent::P
    function ScaledNamedDimsArray{T}(coeff::Number, a::AbstractNamedDimsArray) where {T}
        @assert scaled_eltype(coeff, a) == T begin
            lazy"target type $T cannot hold products of " *
                lazy"$(typeof(coeff)) and $(eltype(a)) objects"
        end
        return new{T, ndims(a), typeof(coeff), eltype(a), typeof(a)}(coeff, a)
    end
end
function ScaledNamedDimsArray(coeff::Number, a::AbstractNamedDimsArray)
    return ScaledNamedDimsArray{scaled_eltype(coeff, a)}(coeff, a)
end
Base.eltype(a::ScaledNamedDimsArray) = scaled_eltype(a.coeff, a.parent)
Base.ndims(a::ScaledNamedDimsArray) = ndims(a.parent)

Base.similar(a::ScaledNamedDimsArray) = similar(a.parent)
Base.similar(a::ScaledNamedDimsArray, elt::Type) = similar(a.parent, elt)
Base.similar(a::ScaledNamedDimsArray, ax) = similar(a.parent, ax)
Base.similar(a::ScaledNamedDimsArray, elt::Type, ax) = similar(a.parent, elt, ax)

using TensorAlgebra: add!
function Base.copyto!(dest::AbstractNamedDimsArray, src::ScaledNamedDimsArray)
    add!(dest, src.parent, src.coeff, false)
    return dest
end
inds(a::ScaledNamedDimsArray) = inds(a.parent)
denamed(a::ScaledNamedDimsArray) = a.coeff *ₗ denamed(a.parent)
Base.axes(a::ScaledNamedDimsArray) = axes(a.parent)
Base.size(a::ScaledNamedDimsArray) = size(a.parent)
TI.iscall(::ScaledNamedDimsArray) = true
TI.operation(::ScaledNamedDimsArray) = *
TI.arguments(a::ScaledNamedDimsArray) = (a.coeff, a.parent)

*ₗ(α::Number, a::ScaledNamedDimsArray) = (α * a.coeff) *ₗ a.parent
conjed(a::ScaledNamedDimsArray) = conj(a.coeff) *ₗ conjed(a.parent)

struct ConjNamedDimsArray{T, N, P <: AbstractNamedDimsArray{T, N}} <:
    AbstractLazyNamedDimsArray{T, N}
    parent::P
end
Base.eltype(a::ConjNamedDimsArray) = eltype(a.parent)
Base.ndims(a::ConjNamedDimsArray) = ndims(a.parent)

Base.similar(a::ConjNamedDimsArray, elt::Type) = similar(a.parent, elt)
Base.similar(a::ConjNamedDimsArray, elt::Type, ax) = similar(a.parent, elt, ax)

using TensorAlgebra: add!
function Base.copyto!(dest::AbstractNamedDimsArray, src::ConjNamedDimsArray)
    add!(dest, src, true, false)
    return dest
end
inds(a::ConjNamedDimsArray) = inds(a.parent)
denamed(a::ConjNamedDimsArray) = conjed(denamed(a.parent))
Base.axes(a::ConjNamedDimsArray) = axes(a.parent)
Base.size(a::ConjNamedDimsArray) = size(a.parent)
TI.iscall(::ConjNamedDimsArray) = true
TI.operation(::ConjNamedDimsArray) = conj
TI.arguments(a::ConjNamedDimsArray) = (a.parent,)

Base.conj(a::ConjNamedDimsArray) = a.parent
