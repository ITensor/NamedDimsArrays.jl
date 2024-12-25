using SimpleTraits: SimpleTraits, @traitdef, @traitfn, @traitimpl, Not

isnamed(x) = isnamed(typeof(x))
isnamed(::Type) = false

# By default, the name of an object is itself.
name(x) = x

@traitdef IsNamed{X}
#! format: off
@traitimpl IsNamed{X} <- isnamed(X)
#! format: on

@traitfn unname(x::X) where {X; IsNamed{X}} = dename(x)
@traitfn unname(x::X) where {X; !IsNamed{X}} = x
