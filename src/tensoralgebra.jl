using LinearAlgebra: LinearAlgebra, qr
using TensorAlgebra: TensorAlgebra, blockedperm, contract, contract!, fusedims, splitdims
using TensorAlgebra.BaseExtensions: BaseExtensions

function TensorAlgebra.contract!(
  a_dest::AbstractNamedDimsArray,
  a1::AbstractNamedDimsArray,
  a2::AbstractNamedDimsArray,
  α::Number=true,
  β::Number=false,
)
  contract!(
    dename(a_dest),
    dimnames(a_dest),
    dename(a1),
    dimnames(a1),
    dename(a2),
    dimnames(a2),
    α,
    β,
  )
  return a_dest
end

function TensorAlgebra.contract(
  a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray, α::Number=true
)
  a_dest, dimnames_dest = contract(dename(a1), dimnames(a1), dename(a2), dimnames(a2), α)
  return nameddims(a_dest, dimnames_dest)
end

function Base.:*(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
  return contract(a1, a2)
end

function LinearAlgebra.mul!(
  a_dest::AbstractNamedDimsArray,
  a1::AbstractNamedDimsArray,
  a2::AbstractNamedDimsArray,
  α::Number=true,
  β::Number=false,
)
  contract!(a_dest, a1, a2, α, β)
  return a_dest
end

function TensorAlgebra.blockedperm(na::AbstractNamedDimsArray, nameddim_blocks::Tuple...)
  # Extract names if named dimensions or axes were passed
  dimname_blocks = map(group -> name.(group), nameddim_blocks)
  dimnames_a = dimnames(na)
  perms = map(dimname_blocks) do dimname_block
    return BaseExtensions.indexin(dimname_block, dimnames_a)
  end
  return blockedperm(perms...)
end

# i, j, k, l = named.((2, 2, 2, 2), ("i", "j", "k", "l"))
# a = randn(i, j, k, l)
# fusedims(a, (i, k) => "a")
# fusedims(a, (i, k) => "a", (j, l) => "b")
# TODO: Rewrite in terms of `fusedims(a, .., (1, 3))` interface.
function TensorAlgebra.fusedims(na::AbstractNamedDimsArray, fusions::Pair...)
  dimnames_fuse = map(group -> name.(group), first.(fusions))
  dimnames_fused = map(name, last.(fusions))
  if sum(length, dimnames_fuse) < ndims(na)
    # Not all names are specified
    dimnames_unspecified = setdiff(dimnames(na), dimnames_fuse...)
    dimnames_fuse = vcat(tuple.(dimnames_unspecified), collect(dimnames_fuse))
    dimnames_fused = vcat(dimnames_unspecified, collect(dimnames_fused))
  end
  perm = blockedperm(na, dimnames_fuse...)
  a_fused = fusedims(unname(na), perm)
  return nameddims(a_fused, dimnames_fused)
end

function TensorAlgebra.splitdims(na::AbstractNamedDimsArray, splitters::Pair...)
  fused_names = map(name, first.(splitters))
  split_namedlengths = last.(splitters)
  splitters_unnamed = map(splitters) do splitter
    fused_name, split_namedlengths = splitter
    fused_dim = findfirst(isequal(fused_name), dimnames(na))
    split_lengths = unname.(split_namedlengths)
    return fused_dim => split_lengths
  end
  a_split = splitdims(unname(na), splitters_unnamed...)
  names_split = Any[tuple.(dimnames(na))...]
  for splitter in splitters
    fused_name, split_namedlengths = splitter
    fused_dim = findfirst(isequal(fused_name), dimnames(na))
    split_names = name.(split_namedlengths)
    names_split[fused_dim] = split_names
  end
  names_split = reduce((x, y) -> (x..., y...), names_split)
  return nameddims(a_split, names_split)
end

function LinearAlgebra.qr(
  a::AbstractNamedDimsArray, dimnames_codomain, dimnames_domain; positive=nothing
)
  @assert isnothing(positive) || !positive
  # TODO: This should be `TensorAlgebra.qr` rather than overloading `LinearAlgebra.qr`.
  # TODO: Don't require wrapping in `Tuple`.
  q, r = qr(
    unname(a),
    Tuple(dimnames(a)),
    Tuple(name.(dimnames_codomain)),
    Tuple(name.(dimnames_domain)),
  )
  name_qr = randname(dimnames(a)[1])
  dimnames_q = (name.(dimnames_codomain)..., name_qr)
  dimnames_r = (name_qr, name.(dimnames_domain)...)
  return nameddims(q, dimnames_q), nameddims(r, dimnames_r)
end

function LinearAlgebra.qr(a::AbstractNamedDimsArray, dimnames_codomain; kwargs...)
  return qr(a, dimnames_codomain, setdiff(dimnames(a), name.(dimnames_codomain)); kwargs...)
end
