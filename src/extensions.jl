import LinearAlgebra: dot

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
LinearAlgebra.AbstractMatrix(A::AbstractMatrix) = A

############################### Cholesky #######################################
const CholeskyOrPiv{T} = Union{Cholesky{T}, CholeskyPivoted{T}}
const CholOrPiv{T} = CholeskyOrPiv{T}

LinearAlgebra.issuccess(C::CholeskyPivoted) = C.info ≥ 0 # either complete or low rank within tolerance
LinearAlgebra.adjoint(C::CholOrPiv{<:Real}) = C # since hermitian
LinearAlgebra.transpose(C::CholOrPiv{<:Real}) = C # since hermitian
LinearAlgebra.ishermitian(C::Union{Cholesky, CholeskyPivoted}) = true # C.info > 0
LinearAlgebra.issymmetric(C::Cholesky) = eltype(C) <: Real && ishermitian(C)
LinearAlgebra.ishermitian(A::Factorization) = ishermitian(Matrix(A))

function LinearAlgebra.logabsdet(C::Union{Cholesky, CholeskyPivoted})
    logdet(C), one(eltype(C))
end

############################## BunchKaufman ####################################
LinearAlgebra.adjoint(A::BunchKaufman) = A
LinearAlgebra.transpose(A::BunchKaufman) = A

# LDL
LinearAlgebra.ishermitian(A::LDLt) = true
LinearAlgebra.issymmetric(A::LDLt) = eltype(A) <: Real && ishermitian(A)

################################################################################
# is positive semi-definite
function ispsd end
ispsd(A::Number, tol::Real = 0.) = A ≥ -tol
# TODO: replace with own pivoted cholesky
ispsd(A::AbstractMatrix, tol::Real = 0.) = all(A->ispsd(A, tol), eigvals(A))
iscov(A::AbstractMatrix, tol::Real = 0.) = issymmetric(A) && ispsd(A, tol)

######################### Extending LinearAlgebra's dot #######################
# function dot(x::AbstractVecOrMat, A::CholeskyPivoted, y::AbstractVecOrMat)
# 	dot(x, LowRank(A), y)
#     # ip = invperm(A.p)
#     # U = ifelse(A.rank == size(A, 1), A.U , @view A.U[1:A.rank, :])
#     # Ux = U * x[A.p] # not permute!, since this otherwise fails if x is a Gramian
#     # Uy = ifelse(x === y, Ux, U * y[A.p])
#     # dot(Ux, Uy)
# end

# function Base.:*(x::AbstractVecOrMat, A::CholeskyPivoted, y::AbstractVecOrMat)
# 	*(x, LowRank(A), y)
#     # ip = invperm(A.p)
#     # U = ifelse(A.rank == size(A, 1), A.U, @view A.U[1:A.rank, :])
#     # Ux = U * (x[:, A.p])' # take care of rank
#     # Uy = ifelse(x === y', Ux, U * y[A.p,:])
#     # (Ux'Uy)
# end

########## Fast multiplication with low-rank pivoted qr and cholesky ###########
# Base.:*(F::QRPivoted, x::AbstractVecOrMat) = (F.Q * F.R)[:,invperm(F.p)]
# function Base.:*(F::CholeskyPivoted, x::AbstractVecOrMat)
#     ip = invperm(F.p)
#     F.L[ip, 1:F.rank] * (F.U[1:F.rank, ip] * x)
# end
# Base.:*(x::AbstractMatrix, F::CholeskyPivoted) = (F*x')'

# these two might not be a good idea because dense multiplication is faster
# function dot(x::AbstractVecOrMat, A::Cholesky, y::AbstractVecOrMat)
#     x ≡ y ? sum(abs2, A.U * y) : dot(x'A.L, A.U * y)
# end

# function Base.:*(x::AbstractVecOrMat, A::Cholesky, y::AbstractVecOrMat)
#     if x === y'
#         Uy = A.U * y
#         Uy'Uy
#     else
#         (x * A.L) * (A.U * y)
#     end
# end
