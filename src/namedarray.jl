struct NamedArray{T,N,Value<:AbstractArray{T,N},Name} <:
       AbstractNamedArray{NamedInteger{T,Name},N,Value,Name}
  value::Value
  name::Name
end

# Minimal interface.
dename(a::NamedArray) = a.value
name(a::NamedArray) = a.name
