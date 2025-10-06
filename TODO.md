- Define `@align`/`@aligned` such that:
```julia
i = namedoneto(2, "i")
j = namedoneto(2, "j")
a = randn(i, j)
@align a[j, i]
@aligned a[j, i]
```
aligns the dimensions (currently `a[j, i]` doesn't align the dimensions).
It could be written in terms of `align_getindex`/`align_view`.
- `svd`, `eigen` (including tensor versions)
- `reshape`, `vec`, including fused dimension names.
- Dimension name set logic, i.e. `setdiffinds(a::AbstractNamedDimsArray, b::AbstractNamedDimsArray)`, etc.
- `swapinds` (written in terms of `mapinds`/`replaceinds`).
- `mapinds(f, a::AbstractNamedDimsArray)` (rename `replaceinds(f, a)` to `mapinds(f, a)`, or have both?)
- `cat` (define `CatName` as a combination of the input names?).
- `canonize`/`flatten_array_wrappers` (https://github.com/mcabbott/NamedPlus.jl/blob/v0.0.5/src/permute.jl#L207)
  - `nameddims(PermutedDimsArray(a, perm), inds)` -> `nameddims(a, inds[invperm(perm)])`
  - `nameddims(transpose(a), inds)` -> `nameddims(a, reverse(inds))`
  - `Transpose(nameddims(a, inds))` -> `nameddims(a, reverse(inds))`
  - etc.
- `MappedName(old_name, name)`, acts like `Name(name)` but keeps track of the old name.
  - `nameddimsmap(a, ::Pair...)`: `namedmap(named(randn(2, 2, 2, 2), i, j, k, l), i => k, j => l)`
    represents that the names map back and forth to each other for the sake of `transpose`,
    `tr`, `eigen`, etc. Operators are generally `namedmap(named(randn(2, 2), i, i'), i => i')`.
- `prime(:i) = PrimedName(:i)`, `prime(:i, 2) = PrimedName(:i, 2)`, `prime(prime(:i)) = PrimedName(:i, 2)`,
  `Name(:i)' = prime(:i)`, etc.
    - Also `prime(f, a::AbstractNamedDimsArray)` where `f` is a filter function to determine
      which dimensions to filter.
- `transpose`/`adjoint` based on `swapinds` and `MappedName(old_name, new_name)`.
  - `adjoint` could make use of a lazy `ConjArray`.
  - `transpose(a, dimname1 => dimname1′, dimname2 => dimname2′)` like `https://github.com/mcabbott/NamedPlus.jl`.
    - Same as `replacedims(a, dimname1 => dimname1′, dimname1′ => dimname1, dimname2 => dimname2′, dimname2′ => dimname2)`.
  - `transpose(f, a)` like the function form of `replace`.
- `tr` based on `MappedName(old_name, name)`.
- Slicing: `nameddims(a, "i", "j")[1:2, 1:2] = nameddims(a[1:2, 1:2], Name(named(1:2, "i")), Name(named(1:2, "j")))`, i.e.
  the parent gets sliced and the new dimensions names are the named slice.
  - Should `NamedDimsArray` store the named axes rather than just the dimension names?
  - Should `NamedDimsArray` have special axes types so that `axes(nameddims(a, "i", "j")) == axes(nameddims(a', "j", "i"))`,
    i.e. equality is based on `issetequal` and not dependent on the ordering of the dimensions?
