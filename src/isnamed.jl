using SimpleTraits: SimpleTraits, @traitdef, @traitfn, @traitimpl, Not

isnamed(x) = isnamed(typeof(x))
isnamed(::Type) = false

# By default, the name of an object is itself.
name(x) = x
nametype(type::Type) = type

@traitdef IsNamed{X}
#! format: off
@traitimpl IsNamed{X} <- isnamed(X)
#! format: on
