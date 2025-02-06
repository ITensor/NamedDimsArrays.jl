module NamedDimsArraysGradedUnitRangesExt

using GradedUnitRanges: GradedUnitRanges, dual, isdual
using NamedDimsArrays: AbstractNamedUnitRange, dename, name, named

function GradedUnitRanges.dual(r::AbstractNamedUnitRange)
  return named(dual(dename(r)), dual(name(r)))
end

function GradedUnitRanges.isdual(r::AbstractNamedUnitRange)
  return isdual(dename(r))
end

end
