using Random: Random, AbstractRNG, randstring

# Generate a new random name, for example in matrix
# factorizations.
randname(rng::AbstractRNG, type::Type) = rand(rng, type)

randname(name; kwargs...) = randname(Random.default_rng(), name; kwargs...)
randname(rng::AbstractRNG, name; kwargs...) = randname(rng, typeof(name); kwargs...)

randname(rng::AbstractRNG, ::Type{<:AbstractString}; length=8) = randstring(rng, length)
function randname(rng::AbstractRNG, ::Type{Symbol}; kwargs...)
  return Symbol(randname(rng, String; kwargs...))
end
