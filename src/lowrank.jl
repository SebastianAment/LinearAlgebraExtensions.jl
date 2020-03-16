############################# Low Rank #########################################
# Low rank factorization
# A = UV
# Separated?
struct LowRank{T, M, N} <: Factorization{T}
    U::M
    V::N
    # rank::Int # rank of the factorization
    tol::T # tolerance / error bound in trace norm
    info::Int # can hold error information about factorization
    function LowRank(U::AbstractVecOrMatOrFac, V::AbstractMatOrFac = U';
                    tol = 1e2eps(eltype(U)), info::Int = 0)
        T = promote_type(eltype(U), eltype(V))
        rank = size(U, 2) == size(V, 1) ? size(U, 2) : throw(
            DimensionMismatch("U and V do not have compatible inner dimensions"))
        new{T, typeof(U), typeof(V)}(U, V, tol, info)
    end
end
const Separated = LowRank # aka separated factorization
Base.:+(A::LowRank, B::LowRank) = LowRank(hcat(A.U, B.U), vcat(A.V, B.V))
function Base.:+(A::LowRank, B::Adjoint{<:Number, <:LowRank})
    LowRank(hcat(A.U, B.parent.V'), vcat(A.V, B.parent.U'))
end
Base.:+(A::Adjoint{<:Number, <:LowRank}, B::LowRank) = B + A
Base.sum(L::LowRank) = sum(L.U) * sum(L.V)

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
# TODO move this to ==, instead of ===, compare speed in case pointers are equal
LinearAlgebra.issymmetric(L::LowRank) = L.U ≡ transpose(L.V)
LinearAlgebra.ishermitian(L::LowRank) = L.U ≡ L.V'
Base.eltype(L::LowRank{T}) where T = T
function LinearAlgebra.tr(L::LowRank)
    n = checksquare(L)
    t = zero(eltype(L))
    for i in 1:n
        for k in 1:rank(L)
            t += L.U[i,k] * L.V[k,i]
        end
    end
    return t
end

# TODO: take care of multiplication with adjoints of LowRank!
Base.:*(L::LowRank, x::AbstractVector) = L.U*(L.V*x)
Base.:*(x::AbstractVector, L::LowRank) = (x*L.U)*L.V
Base.:*(L::LowRank, A::AbstractMatOrFac) = LowRank(L.U, L.V*A)
Base.:*(A::AbstractMatOrFac, L::LowRank) = LowRank(A*L.U, L.V)

Base.:*(Inv::Union{Inverse, PseudoInverse}, L::LowRank) = LowRank(Inv * L.U, L.V)
Base.:*(L::LowRank, Inv::Union{Inverse, PseudoInverse}) = LowRank(L.U, L.V * Inv)

function Base.:*(A::LowRank, B::LowRank)
    C = A.V * B.U
    size(C, 1) ≥ size(C, 2) ? LowRank(A.U*C, B.V) : LowRank(A.U, C*B.V)
end
Base.:\(A::AbstractMatOrFac, L::LowRank) = LowRank(A \ L.U, L.V)
Base.:/(L::LowRank, A::AbstractMatOrFac) = LowRank(L.U, L.V / A)
# least squares solve
# TODO: make possible for matrix
Base.:\(L::LowRank, b::AbstractVector) = pinverse(L.V) * (pinverse(L.U) * b)
Base.:/(b::AbstractVector, L::LowRank) = (b * pinverse(L.V, :R)) * pinverse(L.U, :R)

# ternary operations
# we could also write this with a lazy Matrix-Vector product
function dot(X::AbstractVecOrMatOrFac, S::LowRank, Y::AbstractVecOrMatOrFac)
    UX = S.U'X
    VY = (ishermitian(S) && X ≡ Y) ? UX : S.V * Y
    dot(UX, VY)
end

function Base.:*(X::AbstractMatOrFac, S::LowRank, Y::AbstractVecOrMatOrFac)
    XU = X'S.U
    VY = (ishermitian(S) && X ≡ Y') ? XU' : S.V * Y
    LowRank(XU, VY)
end
# Base.:/(L::LowRank, x::AbstractVector) = LowRank(L.U, L.V / A)

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
