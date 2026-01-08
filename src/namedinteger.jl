struct NamedInteger{Value <: Integer, Name} <: AbstractNamedInteger{Value, Name}
    value::Value
    name::Name
end

# Minimal interface.
denamed(i::NamedInteger) = i.value
name(i::NamedInteger) = i.name
