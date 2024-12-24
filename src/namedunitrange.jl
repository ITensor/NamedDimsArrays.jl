struct NamedUnitRange{T,Value<:AbstractUnitRange{T},Name} <:
       AbstractNamedUnitRange{NamedInteger{T,Name},Value,Name}
  value::Value
  name::Name
end

# Minimal interface.
dename(i::NamedUnitRange) = i.value
name(i::NamedUnitRange) = i.name
