# TODO: could have type which lazily creates views,
# and has access to the underlying matrix
# converts a matrix to a vector, where each element is a
# view into a column of the original matrix
function vecofvec(A::AbstractMatrix)
    [view(A, :, i) for i in 1:size(A, 2)] # one view allocates 64 bytes
end
function matrix(v::AbstractVector{<:AbstractVector})
    n = length(v)
    m = length(v[1])
    all(x -> length(x) == m, v) || error("all element vectors have to have the same
                                            length to be converted to a matrix")
    A = Matrix{eltype(v[1])}(undef, m, n)
    for i in 1:n
        @. A[:, i] = v[i]
    end
    return A
end

# struct DataVector{T, V, AT} <: AbstractVector{V}
#     A::AT
#     function DataVector(A::AbstractMatrix)
#         T = eltype(A)
#         V = typeof(@view(V.A[:,1]))
#         new{T, V, AT}
#     end
# end
# Base.length(V::DataVector) = size(V.A, 2)
# function Base.getindex(V::DataVector, i::Integer, lazy::Val{true} = Val(true))
#     return @view(V.A[:,i])
# end
# # function Base.getindex(V::DataVector, i::Integer, lazy::Val{false})
# #     return V.A[:,i]
# # end
# Base.Matrix(V::DataVector) = V.A
