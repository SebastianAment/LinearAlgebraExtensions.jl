using LinearAlgebra
using LazyInverse: pseudoinverse, PseudoInverse

########################### Projection Matrix ##################################
# stands for A*(AF\y) = A*inverse(A'A)*(A'y) = A*pseudoinverse(A)
struct Projection{T, AT<:AbstractMatOrFac{T},
                                AFT<:AbstractMatOrFac{T}} <: Factorization{T}
    A::AT
    A⁺::AFT
    # temp::V             V<:AbstractVecOrMat
end
Projection(A::AbstractMatOrFac) = Projection(A, pseudoinverse(qr(A, Val(true)))) # defaults to pivoted qr

# TODO: potentially do memory pre-allocation (including temporary)
(P::Projection)(x::AbstractVecOrMatOrFac) = P.A * (P.A⁺ * x)

Base.size(P::Projection, k::Integer) = 0 < k ≤ 2 ? size(P.A, 1) : 1
Base.size(P::Projection) = (size(P, 1), size(P, 2))

# properties
LinearAlgebra.Matrix(P::Projection) = Matrix(P.A * P.A⁺)
LinearAlgebra.adjoint(P::Projection) = P
Base.:^(P::Projection, n::Integer) = P
function Base.literal_pow(::typeof(^), P::Projection, ::Val{N}) where N
    N > 0 ? P : error("Projection P is not invertible")
end
Base.:*(P::Projection, x::AbstractVecOrMat) = P(x)
Base.:*(x::AbstractVecOrMat, P::Projection) = P(x')'

############################# Low Rank #########################################
# Low rank factorization
# A = UV
# Separated?
# TODO: least squares solve vie Projection + PseudoInverse?
struct LowRank{T, M<:AbstractMatOrFac, N<:AbstractMatOrFac} <: Factorization{T}
    U::M
    V::N
    # rank::Int # rank of the factorization
    tol::T # tolerance / error bound in trace norm
    info::Int # can hold error information about factorization
    function LowRank(U::AbstractVecOrMatOrFac, V::AbstractVecOrMatOrFac,
                    tol = 1e2eps(eltype(U)), info::Int = 0)
        T = promote_type(eltype(U), eltype(V))
        rank = size(U, 2) == size(V, 1) ? size(U, 2) : throw(
            DimensionMismatch("U and V do not have compatible inner dimensions"))
        new{T, typeof(U), typeof(V)}(U, V, tol, info)
    end
end

LowRank(u::AbstractVector) = LowRank(reshape(u, :, 1), u')
function LowRank(U::AbstractMatOrFac, tol::Real = eps(eltype(U)), info::Int = 0)
    LowRank(U, U', tol, info)
end

# should only be used if C.rank < size(C, 1)
function LowRank(C::CholeskyPivoted{T}) where {T}
    ip = invperm(C.p)
    U = C.U[1:C.rank, ip]
    LowRank(U', C.tol, C.info)
end

Base.size(S::LowRank) = (size(S.U, 1), size(S.V, 2))
LinearAlgebra.rank(S::LowRank) = size(S.U, 2)
LinearAlgebra.issuccess(S::LowRank) = S.info ≥ 0
Base.Matrix(S::LowRank) = S.U*S.V

import LinearAlgebra: dot, *, \, /, adjoint
# adjoint(S::LowRank) = Adjoint(S)
LinearAlgebra.issymmetric(S::LowRank) = S.U ≡ transpose(S.V)
LinearAlgebra.ishermitian(S::LowRank) = S.U ≡ S.V'

Base.:*(S::LowRank, A::AbstractVecOrMat) = S.U*(S.V*A)
Base.:*(A::AbstractVecOrMat, S::LowRank) = (S*A')'

# ternary operations
# we could also write this with a lazy Matrix-Vector product
function dot(X::AbstractVecOrMatOrFac, S::LowRank, Y::AbstractVecOrMatOrFac)
    UX = S.U'X
    VY = (ishermitian(S) && X ≡ Y) ? UX : S.V * Y
    dot(UX, VY)
end

function *(X::AbstractMatOrFac, S::LowRank, Y::AbstractVecOrMatOrFac)
    XU = X'S.U
    VY = (ishermitian(S) && X ≡ Y') ? XU' : S.V * Y
    XU*VY # LowRank(XU, VY)
end

# alternating least squares for low rank decomposition
function als!(U::AbstractMatrix, V::AbstractMatrix, A::AbstractMatrix,
                tol::Real = 1e-12, maxiter::Int = 32)
    info = -1
    for i in 1:maxiter
        U .= A / V # in place, rdiv!, ldiv!, and qr!, lq!?
        V .= U \ A # need to project rows of V
        if norm(A-U*V) < tol
            info = 0
            break
        end
    end
    return U, V, info
end
als!(L::LowRank, A::AbstractMatrix, maxiter = 32) = als!(L.U, L.V, A, L.tol, maxiter)

# low rank approximation via als
function lowrank(A::AbstractMatrix{T}, rank::Int, tol::Real = 1e-12,
                maxiter::Int = 32) where {T}
    U = rand(eltype(A), (size(A, 1), rank))
    V = rand(eltype(A), rank, size(A, 2))
    U, V, info = als!(U, V, A, tol, maxiter)
    LowRank(U, V, tol, info)
end

# projected alternating least squares
function pals!(L::LowRank, A::AbstractMatrix, PU::Projection, PV::Projection,
                maxiter::Int = 16)
    for i in 1:maxiter
        L.U .= PU(A/L.V) # in place?
        L.V .= PV((L.U\A)')' # need to project rows of V
    end
    return L
end

############################## Hadamard Product ################################
# TODO: tests and non-allocating versions
# TODO: move this into KroneckerProducts?
struct HadamardProduct{T, A<:Tuple{Vararg{AbstractMatOrFac}}} <: Factorization{T}
    factors::A
    function HadamardProduct(A::Tuple{Vararg{AbstractMatOrFac}})
        T = promote_type(eltype.(A)...)
        all(==(size(A[1]), size.(A))) || error("matrices have to have same size to form Hadamard product")
        HadamardProduct{T, typeof(A)}(A)
    end
end
const SchurProduct = HadamardProduct
# smart constructor
hadamard(A::AbstractMatOrFac...) = HadamardProduct(A)
hadamard(H::HadamardProduct, A::AbstractMatOrFac...) = hadamard(tuple(H.args..., A...))
hadamard(A::AbstractMatOrFac, H::HadamardProduct) = hadamard(tuple(A, H.args...))
const ⊙ = hadamard # \odot

Base.size(H::HadamardProduct) = size(H.factors[1])
Base.getindex(H::HadamardProduct, i, j) = prod(A->A[i, j], H.factors)
Base.eltype(H::HadamardProduct{T}) where {T} = T
function Base.Matrix(H::HadamardProduct)
    [H[i,j] for i in 1:size(H, 1), j in 1:size(H, 2)]
end # could also define AbstractMatrix which could preserve sparsity patterns

# from Horn, Roger A.; Johnson, Charles R. (2012). Matrix analysis. Cambridge University Press.
function Base.:*(H::HadamardProduct, x::AbstractVector)
    checksquare(H) # have to check identity for non-square matrices
    x = Diagonal(x)
    for A in H.factors
        x = (A*x)'
    end
    diag(x)
end
# TODO: check this
inverse(H::HadamardProduct) = hadamard(inverse.(H.factors))
LinearAlgebra.isposdef(H::HadamardProduct) = all(isposdef, H.factors) # this fact makes product kernels p.s.d. (p. 478)
LinearAlgebra.issymmetric(H::HadamardProduct) = all(issymmetric, H.factors)
LinearAlgebra.ishermitian(H::HadamardProduct) = all(ishermitian, H.factors)
LinearAlgebra.issuccess(H::HadamardProduct) = all(issuccess, H.factors)
# TODO: factorize

# function LinearAlgebra.dot(x::AbstractVector, H::Hadamard, y::AbstractVector)
#     tr(Diagonal(x)*H.A*Diagonal(y)*H.B')
# end

############################ Symmetric Rescaling ###############################
# applications for Schur complement, and VerticalRescaling, SoR
# represents D'*A*D
# could be SymmetricRescaling
struct SymmetricRescaling{T, M<:AbstractMatOrFac{T},
                                    N<:AbstractMatOrFac{T}} <: Factorization{T}
    D::M
    A::N
end
import LinearAlgebra: Matrix, *, \, dot, size
Matrix(L::SymmetricRescaling) = L.D' * L.A * L.D

# doing this:
# applied(*, L.D', L.A, L.D)
# wouldn't be the whole story, because we want to define matrix multiplication
# with SymmetricRescaling, leading to efficiency gains with structure

# avoids forming the full matrix
*(L::SymmetricRescaling, B::AbstractVecOrMat) = L.D'*(L.A*(L.D*B))
\(L::SymmetricRescaling, B::AbstractVecOrMat) = L.D'\(L.A\(L.D\B))

function dot(x::T, L::SymmetricRescaling, y::T) where {T<:AbstractArray}
    dot(L.D*x, L.A, L.D*y)
    # memory allocation can be avoided with lazy arrays:
    # Dx = applied(*, L.D, x)
    # Dy = applied(*, L.D, y)
    # dot(Dx, L.A, Dy)
end

# this might be too general to be efficient for the SymmetricRescaling usecase
# struct LazyMatrixProduct{T, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
#     A::M
#     B::M
# end

################
# import LinearAlgebra: +
# function +(S::Symmetric, D::Diagonal)
#     T = copy(S)
#     @inbounds for i in 1:size(D, 1)
#         T.data[i, i] += D[i]
#     end
#     return T
# end
