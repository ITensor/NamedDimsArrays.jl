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
    nameddimsindices(a_dest),
    dename(a1),
    nameddimsindices(a1),
    dename(a2),
    nameddimsindices(a2),
    α,
    β,
  )
  return a_dest
end

function TensorAlgebra.contract(
  a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray, α::Number=true
)
  a_dest, nameddimsindices_dest = contract(
    dename(a1), nameddimsindices(a1), dename(a2), nameddimsindices(a2), α
  )
  return nameddims(a_dest, nameddimsindices_dest)
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
  dimname_blocks = map(group -> to_nameddimsindices(na, group), nameddim_blocks)
  nameddimsindices_a = nameddimsindices(na)
  perms = map(dimname_blocks) do dimname_block
    return BaseExtensions.indexin(dimname_block, nameddimsindices_a)
  end
  return blockedperm(perms...)
end

# i, j, k, l = named.((2, 2, 2, 2), ("i", "j", "k", "l"))
# a = randn(i, j, k, l)
# fusedims(a, (i, k) => "a")
# fusedims(a, (i, k) => "a", (j, l) => "b")
# TODO: Rewrite in terms of `fusedims(a, .., (1, 3))` interface.
function TensorAlgebra.fusedims(na::AbstractNamedDimsArray, fusions::Pair...)
  nameddimsindices_fuse = map(group -> to_nameddimsindices(na, group), first.(fusions))
  nameddimsindices_fused = last.(fusions)
  if sum(length, nameddimsindices_fuse) < ndims(na)
    # Not all names are specified
    nameddimsindices_unspecified = setdiff(nameddimsindices(na), nameddimsindices_fuse...)
    nameddimsindices_fuse = vcat(
      tuple.(nameddimsindices_unspecified), collect(nameddimsindices_fuse)
    )
    nameddimsindices_fused = vcat(
      nameddimsindices_unspecified, collect(nameddimsindices_fused)
    )
  end
  perm = blockedperm(na, nameddimsindices_fuse...)
  a_fused = fusedims(unname(na), perm)
  return nameddims(a_fused, nameddimsindices_fused)
end

function TensorAlgebra.splitdims(na::AbstractNamedDimsArray, splitters::Pair...)
  splitters = to_nameddimsindices(na, first.(splitters)) .=> last.(splitters)
  split_namedlengths = last.(splitters)
  splitters_unnamed = map(splitters) do splitter
    fused_name, split_namedlengths = splitter
    fused_dim = findfirst(isequal(fused_name), nameddimsindices(na))
    split_lengths = unname.(split_namedlengths)
    return fused_dim => split_lengths
  end
  a_split = splitdims(unname(na), splitters_unnamed...)
  names_split = Any[tuple.(nameddimsindices(na))...]
  for splitter in splitters
    fused_name, split_namedlengths = splitter
    fused_dim = findfirst(isequal(fused_name), nameddimsindices(na))
    split_names = name.(split_namedlengths)
    names_split[fused_dim] = split_names
  end
  names_split = reduce((x, y) -> (x..., y...), names_split)
  return nameddims(a_split, names_split)
end

function LinearAlgebra.qr(
  a::AbstractNamedDimsArray,
  nameddimsindices_codomain,
  nameddimsindices_domain;
  positive=nothing,
)
  @assert isnothing(positive) || !positive
  # TODO: This should be `TensorAlgebra.qr` rather than overloading `LinearAlgebra.qr`.
  # TODO: Don't require wrapping in `Tuple`.
  q, r = qr(
    unname(a),
    Tuple(nameddimsindices(a)),
    Tuple(to_nameddimsindices(a, nameddimsindices_codomain)),
    Tuple(to_nameddimsindices(a, nameddimsindices_domain)),
  )
  name_qr = randname(nameddimsindices(a)[1])
  nameddimsindices_q = (to_nameddimsindices(a, nameddimsindices_codomain)..., name_qr)
  nameddimsindices_r = (name_qr, to_nameddimsindices(a, nameddimsindices_domain)...)
  return nameddims(q, nameddimsindices_q), nameddims(r, nameddimsindices_r)
end

function LinearAlgebra.qr(a::AbstractNamedDimsArray, nameddimsindices_codomain; kwargs...)
  return qr(
    a,
    nameddimsindices_codomain,
    setdiff(nameddimsindices(a), to_nameddimsindices(a, nameddimsindices_codomain));
    kwargs...,
  )
end
