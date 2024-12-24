struct NamedInteger{Value<:Integer,Name} <: AbstractNamedInteger{Value,Name}
  value::Value
  name::Name
end

# Minimal interface.
dename(i::NamedInteger) = i.value
name(i::NamedInteger) = i.name
