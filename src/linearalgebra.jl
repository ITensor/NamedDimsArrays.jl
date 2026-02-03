import LinearAlgebra as LA

# We overload `LinearAlgebra.norm` because the LinearAlgebra.jl AbstractArray definition
# uses scalar indexing:
# https://github.com/JuliaLang/LinearAlgebra.jl/blob/3a4fdad7f608928ecb4b41e76b1e9ecacd058444/src/generic.jl#L575-L724
# which isn't friendly for NamedDimsArrays wrapping GPU arrays.
# This implicitly helps with defining `LA.normalize[!]` as well (though note that
# uses `LinearAlgebra.rmul!` as well).
function LA.norm(a::AbstractNamedDimsArray, p::Real = 2; kwargs...)
    return LA.norm(denamed(a), p; kwargs...)
end

# We overload these because the LinearAlgebra.jl AbstractArray definitions of `rmul!`,
# `lmul!`, `rdiv!`, and `ldiv!` use scalar indexing:
# https://github.com/JuliaLang/LinearAlgebra.jl/blob/3a4fdad7f608928ecb4b41e76b1e9ecacd058444/src/generic.jl#L266-L395
# which isn't friendly for NamedDimsArrays wrapping GPU arrays.
for f! in [:mul!, :div!]
    lf! = Symbol(:l, f!)
    rf! = Symbol(:r, f!)
    @eval begin
        function LA.$rf!(a::AbstractNamedDimsArray, α::Number)
            LA.$rf!(denamed(a), α)
            return a
        end
        function LA.$lf!(α::Number, a::AbstractNamedDimsArray)
            LA.$lf!(α, denamed(a))
            return a
        end
    end
end

# We overload `LienarAlgebra.dot` because the LinearAlgebra.jl AbstractArray definition
# uses scalar indexing:
# https://github.com/JuliaLang/LinearAlgebra.jl/blob/3a4fdad7f608928ecb4b41e76b1e9ecacd058444/src/generic.jl#L919-L1009
# which isn't friendly for NamedDimsArrays wrapping GPU arrays.
function LA.dot(a1::AbstractNamedDimsArray, a2::AbstractNamedDimsArray)
    # TODO: Should we define:
    # `TensorAlgebra.permdot(denamed(a1), denamed(a2), genperm(dimnames(a1), dimnames(a2)))`
    # in TensorAlgebra.jl and use that here?
    return LA.dot(denamed(a1), denamed(a2, dimnames(a1)))
end
