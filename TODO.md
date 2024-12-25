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
- `MappedName(old_name, name)`, acts like `Name(name)` but keeps track of the old name.
  - `namedmap(a, ::Pair...)`: `namedmap(named(randn(2, 2, 2, 2), i, j, k, l), i => k, j => l)`
    represents that the names map back and forth to each other for the sake of `transpose`,
    `tr`, `eigen`, etc. Operators are generally `namedmap(named(randn(2, 2), i, i'), i => i')`.
- `prime(:i) = PrimedName(:i)`, `prime(:i, 2) = PrimedName(:i, 2)`, `prime(prime(:i)) = PrimedName(:i, 2)`,
  `Name(:i)' = prime(:i)`, etc.
- `transpose`/`adjoint` based on `swapdimnames` and `MappedName(old_name, new_name)`.
  - `adjoint` could make use of a lazy `ConjArray`.
  - `transpose(a, dimname1 => dimname1′, dimname2 => dimname2′)` like `https://github.com/mcabbott/NamedPlus.jl`.
    - Same as `replacedims(a, dimname1 => dimname1′, dimname1′ => dimname1, dimname2 => dimname2′, dimname2′ => dimname2)`.
  - `transpose(f, a)` like the function form of `replace`.
- `tr` based on `MappedName(old_name, name)`.
- Slicing: `nameddims(a, "i", "j")[1:2, 1:2] = nameddims(a[1:2, 1:2], Name(named(1:2, "i")), Name(named(1:2, "j")))`, i.e.
  the parent gets sliced and the new dimensions names are the named slice.
  - Should `NamedDimsArray` store the named axes rather than just the dimension names?
  - Should `NamedDimsArray` have special axes types so that `axes(nameddims(a, "i", "j")) == axes(nameddims(a', "j", "i"))`?
