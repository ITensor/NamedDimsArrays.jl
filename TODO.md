- `svd`, `eigen` (including tensor versions)
- `reshape`, `vec`
- `swapdimnames`
- `mapdimnames(f, a::AbstractNamedDimsArray)` (rename `replacedimnames(f, a)` to `mapdimnames(f, a)`, or have both?)
- `cat` (define `CatName` as a combination of the input names?).
- `canonize`/`flatten_array_wrappers` (https://github.com/mcabbott/NamedPlus.jl/blob/v0.0.5/src/permute.jl#L207)
  - `nameddims(PermutedDimsArray(a, perm), dimnames)` -> `nameddims(a, dimnames[invperm(perm)])`
  - `nameddims(transpose(a), dimnames)` -> `nameddims(a, reverse(dimnames))`
  - `Transpose(nameddims(a, dimnames))` -> `nameddims(a, reverse(dimnames))`
  - etc.
- `MappedName(old_name, new_name)`.
  - `namemap(a, ::Pair...)`: `namemap(named(randn(2, 2, 2, 2), i, j, k, l), i => k, j => l)`
    represents that the names map back and forth to each other for the sake of `transpose`,
    `tr`, `eigen`, etc. Operators are generally `namemap(named(randn(2, 2), i, i'), i => i')`.
- `transpose`/`adjoint` based on `swapdimnames` and `MappedName(old_name, new_name)`.
  - `adjoint` could make use of a lazy `ConjArray`.
  - `transpose(a, dimname1 => dimname1′, dimname2 => dimname2′)` like `https://github.com/mcabbott/NamedPlus.jl`.
    - Same as `replacedims(a, dimname1 => dimname1′, dimname1′ => dimname1, dimname2 => dimname2′, dimname2′ => dimname2)`.
  - `transpose(f, a)` like the function form of `replace`.
- `tr` based on `MappedName(old_name, new_name)`.
- `prime` based on `MappedName(name, PrimedName(name))` (should it alias to `'`?).
- `prime(Name(:i)) = PrimedName(:i)`

```julia
i = Name(:i)
j = Name(:j)
k = Name(:k)
a = randn(2, 2)
b = randn(2, 2)
a[i, j] * b[j, k]
aij = a[i, j]
aij[j[2], i[1]] # aij[named(:j, 2), named(:i, 1)]
aij[j, i] # align(aij, (j, i))
@view aij[j, i] # aligned(aij, (j, i))
```
