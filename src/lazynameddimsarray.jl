import TensorAlgebra as TA
using TensorAlgebra: +ₗ, *ₗ, conjed

copy_lazynameddims(a::AbstractArray) = copyto!(similar(a), a)

TA.@scaledarray_type ScaledNamedDimsArray AbstractNamedDimsArray
TA.@scaledarray ScaledNamedDimsArray AbstractNamedDimsArray
TA.:*ₗ(α::Number, a::AbstractNamedDimsArray) = ScaledNamedDimsArray(α, a)
Base.copy(a::ScaledNamedDimsArray) = copy_lazynameddims(a)
inds(a::ScaledNamedDimsArray) = inds(TA.unscaled(a))
denamed(a::ScaledNamedDimsArray) = coeff(a) *ₗ denamed(TA.unscaled(a))

TA.@conjarray_type ConjNamedDimsArray AbstractNamedDimsArray
TA.@conjarray ConjNamedDimsArray AbstractNamedDimsArray
TA.conjed(a::AbstractNamedDimsArray) = ConjNamedDimsArray(a)
Base.copy(a::ConjNamedDimsArray) = copy_lazynameddims(a)
inds(a::ConjNamedDimsArray) = inds(conjed(a))
denamed(a::ConjNamedDimsArray) = conjed(denamed(conjed(a)))
aligneddims(a::ConjNamedDimsArray, dims) = conjed(aligneddims(conjed(a), dims))

TA.@addarray_type AddNamedDimsArray AbstractNamedDimsArray
TA.@addarray AddNamedDimsArray AbstractNamedDimsArray
TA.:+ₗ(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) = AddNamedDimsArray(a, b)
Base.copy(a::AddNamedDimsArray) = copy_lazynameddims(a)
inds(a::AddNamedDimsArray) = inds(first(TA.addends(a)))
function denamed(a::AddNamedDimsArray)
    a′ = denamed(first(TA.addends(a)))
    for addend in Iterators.rest(TA.addends(a))
        a′ = a′ +ₗ denamed(addend, inds(first(TA.addends(a))))
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
# TODO: Don't convert to `Tuple`?
inds(a::MulNamedDimsArray) = Tuple(symdiff(inds.(TA.factors(a))...))
denamed(a::MulNamedDimsArray) = error("`denamed` is not defined for `MulNamedDimsArray`.")
