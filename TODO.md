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
  - `namemap(a, ::Pair...)`: `namemap(named(randn(2, 2), i, j), i <=> j)`
    represents that the names map back and forth to each other for the sake of `transpose`,
    `tr`, `eigen`, etc. Operators are generally `namemap(named(randn(2, 2), i, i'), i <=> i')`.
- `transpose`/`adjoint` based on `swapdimnames` and `MappedName(old_name, new_name)`.
  - `adjoint` could make use of a lazy `ConjArray`.
- `tr` based on `MappedName(old_name, new_name)`.
- `prime` based on `MappedName(name, PrimedName(name))` (should it alias to `'`?).
