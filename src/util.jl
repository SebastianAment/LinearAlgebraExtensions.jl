# converts a matrix to a vector, where each element is a
# view into a column of the original matrix
function vector²(A::AbstractMatrix)
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
