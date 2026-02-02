import TensorAlgebra as TA
using TensorAlgebra: +ₗ, *ₗ, conjed

copy_lazynameddims(a::AbstractArray) = copyto!(similar(a), a)

TA.@scaledarray_type ScaledNamedDimsArray AbstractNamedDimsArray
TA.@scaledarray ScaledNamedDimsArray AbstractNamedDimsArray
TA.:*ₗ(α::Number, a::AbstractNamedDimsArray) = ScaledNamedDimsArray(α, a)
Base.copy(a::ScaledNamedDimsArray) = copy_lazynameddims(a)
dimnames(a::ScaledNamedDimsArray) = dimnames(TA.unscaled(a))
denamed(a::ScaledNamedDimsArray) = coeff(a) *ₗ denamed(TA.unscaled(a))

TA.@conjarray_type ConjNamedDimsArray AbstractNamedDimsArray
TA.@conjarray ConjNamedDimsArray AbstractNamedDimsArray
TA.conjed(a::AbstractNamedDimsArray) = ConjNamedDimsArray(a)
Base.copy(a::ConjNamedDimsArray) = copy_lazynameddims(a)
dimnames(a::ConjNamedDimsArray) = dimnames(conjed(a))
denamed(a::ConjNamedDimsArray) = conjed(denamed(conjed(a)))
aligneddims(a::ConjNamedDimsArray, dims) = conjed(aligneddims(conjed(a), dims))

TA.@addarray_type AddNamedDimsArray AbstractNamedDimsArray
TA.@addarray AddNamedDimsArray AbstractNamedDimsArray
TA.:+ₗ(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) = AddNamedDimsArray(a, b)
Base.copy(a::AddNamedDimsArray) = copy_lazynameddims(a)
dimnames(a::AddNamedDimsArray) = dimnames(first(TA.addends(a)))
function denamed(a::AddNamedDimsArray)
    a′ = denamed(first(TA.addends(a)))
    for addend in Iterators.rest(TA.addends(a))
        a′ = a′ +ₗ denamed(addend, dimnames(first(TA.addends(a))))
    end
    return a′
end

TA.@mularray_type MulNamedDimsArray AbstractNamedDimsArray
TA.@mularray MulNamedDimsArray AbstractNamedDimsArray
TA.:*ₗ(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) = MulNamedDimsArray(a, b)
Base.copy(a::MulNamedDimsArray) = copy_lazynameddims(a)
TA.mul_ndims(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) = length(TA.mul_axes(a, b))
# TODO: Don't convert to `Tuple`?
TA.mul_axes(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) =
    LittleSet(Tuple(symdiff(axes(a), axes(b))))
# Fix ambiguity error.
function Base.similar(
        a::MulNamedDimsArray, elt::Type,
        inds::Tuple{NamedDimsIndices, Vararg{NamedDimsIndices}}
    )
    return TA.similar_mul(a, elt, inds)
end
# Fix ambiguity error.
function Base.similar(a::MulNamedDimsArray, elt::Type, inds::LittleSet)
    return TA.similar_mul(a, elt, inds)
end
# We overload this since the generic implementation of `inds(::AbstractArray)` calls
# `denamed(a)`, which for performance reasons we don't want to define for
# `MulNamedDimsArray` (maybe we could define it in terms of lazy `TensorAlgebra.contract``,
# i.e. `contracted`, but that isn't defined right now and would be more complicated).
# Note that this implicitly defined `axes(a::MulNamedDimsArray)` since that is defined as
# `inds(a)` by default.
# TODO: Don't convert to `Tuple`?
inds(a::MulNamedDimsArray) = LittleSet(Tuple(symdiff(inds.(TA.factors(a))...)))
# TODO: Don't convert to `Tuple`?
dimnames(a::MulNamedDimsArray) = LittleSet(Tuple(symdiff(dimnames.(TA.factors(a))...)))
denamed(a::MulNamedDimsArray) = error("`denamed` is not defined for `MulNamedDimsArray`.")
