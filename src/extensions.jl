import LinearAlgebra: dot

LinearAlgebra.sqrt(I::UniformScaling) = sqrt(I.λ)*I
LinearAlgebra.diag(x::Number) = x

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
const AbstractVecOrTup{T} = Union{AbstractVector{T}, NTuple{N, T} where N}
const AbstractMatOrUni{T} = Union{AbstractMatrix{T}, UniformScaling{T}}
const AbstractMatOrFacOrUni{T} = Union{AbstractMatOrFac{T}, UniformScaling{T}}
const AbstractVecOrMatOrFac{T} = Union{AbstractVecOrMat{T}, Factorization{T}}

# make factorize work on
# precedence: qr line 403
LinearAlgebra.factorize(x::Union{Number, Factorization, UniformScaling}) = x
LinearAlgebra.adjoint(F::Factorization) = Adjoint(F)

const CholeskyOrPiv{T} = Union{Cholesky{T}, CholeskyPivoted{T}}
const CholOrPiv{T} = CholeskyOrPiv{T}

LinearAlgebra.adjoint(C::CholOrPiv{<:Real}) = C # since hermitian
LinearAlgebra.transpose(C::CholOrPiv{<:Real}) = C # since hermitian

################################################################################
# is positive semi-definite
function ispsd end
ispsd(A::Number, tol::Real = 0.) = A ≥ -tol
# TODO: replace with own pivoted cholesky
ispsd(A::AbstractMatrix, tol::Real = 0.) = all(A->ispsd(A, tol), eigvals(A))
iscov(A::AbstractMatrix, tol::Real = 0.) = issymmetric(A) && ispsd(A, tol)

######################### Extending LinearAlgebra's dot #######################
LinearAlgebra.dot(x, A::UniformScaling, y) = dot(x, y) * A.λ

# TODO: The first two definitions will be deprecated in Julia 1.4
function dot(x::AbstractVecOrMat, A::AbstractMatrix, y::AbstractVecOrMat)
    n = LinearAlgebra.checksquare(A)
    length(x) ≠ length(y) && throw(DimensionMismatch("x and y do not have compatible dimensions"))
    n ≠ length(x) && throw(DimensionMismatch("x and A do not have compatible dimensions"))
    s = 0.
    @inbounds for j in 1:size(A, 2)
        s_j = 0.
        @simd for i in 1:size(A, 1)
             s_j += (x[i] * A[i, j])
        end
        s += s_j * y[j]
    end
    return s
end

function dot(x::AbstractVecOrMat, A::Diagonal, y::AbstractVecOrMat)
    n = LinearAlgebra.checksquare(A)
    length(x) ≠ length(y) && throw(DimensionMismatch("x and y do not have compatible dimensions"))
    n ≠ length(x) && throw(DimensionMismatch("x and A do not have compatible dimensions"))
    d = 0
    @inbounds @simd for i in 1:n
        d += x[i] * A[i,i] * y[i]
    end
    d
end

function dot(x::AbstractVecOrMat, A::Cholesky, y::AbstractVecOrMat)
    ifelse(x === y, sum(abs2, A.U * y), dot(x * A.L, A.U * y))
end

function Base.:*(x::AbstractVecOrMat, A::Cholesky, y::AbstractVecOrMat)
    if x === y'
        Uy = A.U * y
        Uy'Uy
    else
        (x * A.L) * (A.U * y)
    end
end

function dot(x::AbstractVecOrMat, A::CholeskyPivoted, y::AbstractVecOrMat)
    ip = invperm(A.p)
    U = ifelse(A.rank == size(A, 1), A.U , @view A.U[1:A.rank, :])
    Ux = U * x[A.p] # not permute!, since this otherwise fails if x is a Gramian
    Uy = ifelse(x === y, Ux, U * y[A.p])
    dot(Ux, Uy)
end

function Base.:*(x::AbstractVecOrMat, A::CholeskyPivoted, y::AbstractVecOrMat)
    ip = invperm(A.p)
    U = ifelse(A.rank == size(A, 1), A.U, @view A.U[1:A.rank, :])
    Ux = U * (x[:, A.p])' # take care of rank
    Uy = ifelse(x === y', Ux, U * y[A.p,:])
    (Ux'Uy)
end

########## Fast multiplication with low-rank pivoted qr and cholesky ###########
# Base.:*(F::QRPivoted, x::AbstractVecOrMat) = (F.Q * F.R)[:,invperm(F.p)]
function Base.:*(F::CholeskyPivoted, x::AbstractVecOrMat)
    ip = invperm(F.p)
    F.L[ip, 1:F.rank] * (F.U[1:F.rank, ip] * x)
end
Base.:*(x::AbstractMatrix, F::CholeskyPivoted) = (F*x')'
