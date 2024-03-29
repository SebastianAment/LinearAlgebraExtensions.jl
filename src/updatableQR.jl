# TODO: move to linear algebra extensions

# code below adapted from https://github.com/oxfordcontrol/GeneralQP.jl/blob/master/src/linear_algebra.jl
using LinearAlgebra
# only needs to be mutable because of views
mutable struct UpdatableQR{T} <: Factorization{T}
    """
    Gives the qr factorization an (n, m) matrix as Q1*R1
    Q2 is such that Q := [Q1 Q2] is orthogonal and R is an (n, n) matrix where R1 "views into".
    """
    Q::Matrix{T}
    R::Matrix{T}
    n::Int
    m::Int

    Q1::SubArray{T, 2, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
    Q2::SubArray{T, 2, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
    R1::UpperTriangular{T, SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int},UnitRange{Int}}, false}}

    function UpdatableQR(A::AbstractMatrix{T}) where {T}
        n, m = size(A)
        m <= n || error("m = $m > $n = n")
        F = qr(A)
        Q = F.Q*Matrix(I, n, n)
        R = zeros(T, n, n)
        @. R[1:m, 1:m] = F.R
        Q1, Q2, R1 = @views Q[:, 1:m], Q[:, m+1:n], UpperTriangular(R[1:m, 1:m])
        new{T}(Q, R, n, m, Q1, Q2, R1)
    end
end
const UQR = UpdatableQR

# add column at specific index
# add it to R, use givens rotations to make
# R upper-triangular, and adjust Q accordingly
function add_column!(F::UpdatableQR, x::AbstractVector, k::Int = size(F, 2)+1)
    n, m = size(F)
    n > m || throw("cannot add column to factorization with maximum rank $n")
    k ≤ m + 1 || throw("cannot add at column index $k to factorization of size $(size(F))")

    # inserting new column into R
    for j in m:-1:k
        @simd for i in 1:j+1
            @inbounds F.R[i, j+1] = F.R[i, j]
        end
    end
    mul!(@view(F.R[:, k]), F.Q', x) # Q' * x

    # update
    F.m += 1; update_views!(F)
    n, m = size(F)

    # zero out the "spike"
    R2 = @view F.R[k:n, k:m] # if we don't restrict, we lose performance
    Q2 = @view F.Q[:, k:end]
    for i in n-(k-1):-1:2
        G, r = givens(R2[i-1, 1], R2[i, 1], i-1, i)
        lmul!(G, R2)
        lmul!(G, Q2')
    end
    return F
end

function remove_column!(F::UpdatableQR, k::Int = size(F, 2))
    1 <= k <= F.m || throw("index $k not in range [1, $(F.m)]")
    Q12, R12 = @views F.Q[:, k:F.m], F.R[k:F.m, k+1:F.m]
    for i in 1:size(R12, 1)-1
        G, r = givens(R12[i, i], R12[i + 1, i], i, i+1)
        lmul!(G, R12)
        lmul!(G, Q12')
    end

    for i in 1:F.m, j in k:F.m-1
        F.R[i, j] = F.R[i, j+1]
    end
    F.R[:, F.m] .= zero(eltype(F))

    F.m -= 1; update_views!(F)
    return F
end

function update_views!(F::UpdatableQR)
    # F.R1 = UpperTriangular(view(F.R, 1:F.m, 1:F.m))
    # F.Q1 = view(F.Q, :, 1:F.m)
    # F.Q2 = view(F.Q, :, F.m+1:F.n)
    n, m = size(F)
    F.Q1, F.Q2 = @views F.Q[:, 1:m], F.Q[:, m+1:end]
    F.R1 = @views UpperTriangular(F.R[1:m, 1:m])
end

######################
Base.Matrix(F::UpdatableQR) = F.Q1*F.R1
Base.AbstractMatrix(F::UpdatableQR) = Matrix(F)
function Base.:\(F::UpdatableQR, X::AbstractVecOrMat)
    Y = F.Q1' * X
    ldiv!(F.R1, Y)
end
####### ldiv!
function LinearAlgebra.ldiv!(Y::AbstractVector, F::UpdatableQR, X::AbstractVector)
    _ldiv_helper!(Y, F, X)
end
function LinearAlgebra.ldiv!(Y::AbstractMatrix, F::UpdatableQR, X::AbstractMatrix)
    _ldiv_helper!(Y, F, X)
end
@inline function _ldiv_helper!(Y, F, X)
    mul!(Y, F.Q1', X)
    ldiv!(F.R1, Y)
end

####### mul!
# WARNING: overwrites X too!
# to make this less dangerous, could add temporary pointer as optional argument
function LinearAlgebra.mul!(Y::AbstractVector, F::UpdatableQR, X::AbstractVector)
    _mul_helper!(Y, F, X)
end
function LinearAlgebra.mul!(Y::AbstractMatrix, F::UpdatableQR, X::AbstractMatrix)
    _mul_helper!(Y, F, X)
end
@inline function _mul_helper!(Y, F, X)
    mul!(X, F.R1, X) # this works because R is triangular
    mul!(Y, F.Q1, X)
end

Base.eltype(F::UpdatableQR{T}) where {T} = T
Base.size(F::UpdatableQR) = (F.n, F.m)
function Base.size(F::UpdatableQR, i::Int)
    if i == 1
        F.n
    elseif i == 2
        F.m
    elseif i > 2
        1
    end
end

function Base.copy(src::UpdatableQR)
    q = @view(src.Q[:, 1])
    q = reshape(q, :, 1)
    dst = UpdatableQR(q)
    copy!(dst, src)
end

function Base.copy!(dst::UpdatableQR, src::UpdatableQR)
    dst.Q .= src.Q
    dst.R .= src.R
    dst.n = src.n
    dst.m = src.m
    update_views!(dst)
    dst
end

################################################################################
# APx = y where A = QR and P is permutation
# avoids memory movement when adding columns at small indices
# INFO: 2 sources of allocation: push! and invpermute!
struct PermutedUpdatableQR{T, FT<:UQR{T}} <: Factorization{T}
    uqr::FT # factorization
    perm::Vector{Int} # permutation of input
    # invperm::Vector{Int}
end
const PUQR = PermutedUpdatableQR
function PermutedUpdatableQR(A::AbstractMatrix, perm = collect(1:size(A, 2)))
    PermutedUpdatableQR(UQR(A), perm)
end
function PermutedUpdatableQR(A::UQR, perm = collect(1:size(A, 2)))
    PermutedUpdatableQR(A, perm)
end

Base.eltype(F::PUQR{T}) where {T} = T
Base.size(F::PUQR) = size(F.uqr)
Base.size(F::PUQR, i::Int) = size(F.uqr, i)
Base.Matrix(F::PUQR) = Matrix(F.uqr)[:, invperm(F.perm)]
Base.AbstractMatrix(F::PUQR) = Matrix(F)

# passing add and remove column to UQR
# always adds to the end and keeps track of permutations in F.perm
function add_column!(F::PUQR, x::AbstractVector, k::Int = size(F, 2)+1)
    for (i, p) in enumerate(F.perm)
        if p ≥ k # incrementing indices above k
            F.perm[i] = p+1
        end
    end
    push!(F.perm, k) # could pre-allocate perm and adjust
    add_column!(F.uqr, x)
end
function remove_column!(F::PUQR, k::Int = size(F, 2))
    i = findfirst(==(k), F.perm)
    # TODO: check i for nothing
    deleteat!(F.perm, i)
    for (i, p) in enumerate(F.perm)
        if p ≥ k # decrementing indices above k
            F.perm[i] = p-1
        end
    end
    remove_column!(F.uqr, i)
end

function Base.:\(P::PUQR, x::AbstractVecOrMat)
    n, m = size(P)
    y = zeros(eltype(P), m)
    ldiv!(y, P, x)
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::PUQR, x::AbstractVector)
    ldiv!(y, P.uqr, x)
    invpermute!(y, P.perm) # I believe this still allocates the invperm
    #invpermute!!(y, P.perm) # this changes P.perm!
end
