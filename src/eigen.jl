# extensions for eigenvalue decomposition
# ldiv! and mul! would be equivalent except for eigenvalues ...
function LinearAlgebra.ldiv!(y::AbstractVector, E::Eigen, x::AbstractVector)
     ldiv!!(y, E, copy(x))
end
# WARNING: uses x as a temporary
function ldiv!!(y::AbstractVector, E::Eigen, x::AbstractVector) # temporary
    mul!(y, E.vectors', x)
    @. x = y / E.values
    mul!(y, E.vectors, x)
end
function LinearAlgebra.mul!(y::AbstractVector, E::Eigen, x::AbstractVector)
     mul!!(y, E, copy(x))
end
function mul!!(y::AbstractVector, E::Eigen, x::AbstractVector) # temporary
    mul!(y, E.vectors', x)
    @. x = y * E.values
    mul!(y, E.vectors, x)
end

LinearAlgebra.:(\)(E::Eigen, x::AbstractVector) = ldiv!(similar(x), E, x)
LinearAlgebra.:(*)(E::Eigen, x::AbstractVector) = mul!(similar(x), E, x)

LazyInverse.inverse(E::Eigen) = inverse!(deepcopy(E))
function inverse!(E::Eigen)
    @. E.values = inv(E.values)
    return E
end
