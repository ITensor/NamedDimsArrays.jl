# Define types.
@scaledarray_type ScaledNamedDimsArray AbstractNamedDimsArray
@scaledarray ScaledNamedDimsArray AbstractNamedDimsArray
@conjarray_type ConjNamedDimsArray AbstractNamedDimsArray
@conjarray ConjNamedDimsArray AbstractNamedDimsArray
@addarray_type AddNamedDimsArray AbstractNamedDimsArray
@addarray AddNamedDimsArray AbstractNamedDimsArray
@mularray_type MulNamedDimsArray AbstractNamedDimsArray
@mularray MulNamedDimsArray AbstractNamedDimsArray

*ₗ(α::Number, a::AbstractNamedDimsArray) = ScaledNamedDimsArray(α, a)
conjed(a::AbstractNamedDimsArray) = ConjNamedDimsArray(a)
+ₗ(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) = AddNamedDimsArray(a, b)
*ₗ(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) = MulNamedDimsArray(a, b)

copy_lazynameddims(a::AbstractArray) = copyto!(similar(a), a)
Base.copy(a::ScaledNamedDimsArray) = copy_lazynameddims(a)
Base.copy(a::ConjNamedDimsArray) = copy_lazynameddims(a)
Base.copy(a::AddNamedDimsArray) = copy_lazynameddims(a)
Base.copy(a::MulNamedDimsArray) = copy_lazynameddims(a)

mul_ndims(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) = length(mul_axes(a, b))
# TODO: Don't convert to `Tuple`?
mul_axes(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray) =
    NaiveOrderedSet(Tuple(symdiff(axes(a), axes(b))))

# Fix ambiguity error.
function Base.similar(
        a::MulNamedDimsArray, elt::Type,
        inds::Tuple{NamedDimsIndices, Vararg{NamedDimsIndices}}
    )
    return similar_mul(a, elt, inds)
end

inds(a::ScaledNamedDimsArray) = inds(unscaled(a))
denamed(a::ScaledNamedDimsArray) = coeff(a) *ₗ denamed(unscaled(a))
inds(a::ConjNamedDimsArray) = inds(conjed(a))
denamed(a::ConjNamedDimsArray) = conjed(denamed(conjed(a)))
inds(a::AddNamedDimsArray) = inds(first(addends(a)))
function denamed(a::AddNamedDimsArray)
    a′ = denamed(first(addends(a)))
    for addend in Iterators.rest(addends(a))
        a′ = a′ +ₗ denamed(addend, inds(first(addends(a))))
    end
    return a′
end
# TODO: Don't convert to `Tuple`?
inds(a::MulNamedDimsArray) = Tuple(symdiff(inds.(factors(a))...))
denamed(a::MulNamedDimsArray) = error("`denamed` is not defined for `MulNamedDimsArray`.")
