############################# Low Rank #########################################
# Low rank factorization
# A = UV
# Separated?
struct LowRank{T, M, N} <: Factorization{T}
    U::M
    V::N
    tol::T # tolerance / error bound in trace norm
    info::Int # can hold error information about factorization
    function LowRank(U::AbstractVecOrMatOrFac, V::AbstractMatOrFac = U';
                    tol = 0., info::Int = 0)
        T = promote_type(eltype(U), eltype(V))
        rank = size(U, 2) == size(V, 1) ? size(U, 2) : throw(
            DimensionMismatch("U and V do not have compatible inner dimensions"))
        rank ≥ 1 || throw("rank = 0")
        new{T, typeof(U), typeof(V)}(U, V, tol, info)
    end
end
const Separated = LowRank # aka separated factorization
function LowRank(init, n::Int, k::Int, m::Int)
    U = init(n, k)
    V = init(k, m)
    LowRank(U, V)
end

# should only be used if C.rank < size(C, 1)
function LowRank(C::CholeskyPivoted{T}) where {T}
    ip = invperm(C.p)
    U = C.U[1:C.rank, ip]
    LowRank(U', C.tol, C.info)
end

Base.:+(A::LowRank, B::LowRank) = LowRank(hcat(A.U, B.U), vcat(A.V, B.V))
function Base.:+(A::LowRank, B::Adjoint{<:Number, <:LowRank})
    LowRank(hcat(A.U, B.parent.V'), vcat(A.V, B.parent.U'))
end
Base.:+(A::Adjoint{<:Number, <:LowRank}, B::LowRank) = B + A
Base.sum(L::LowRank) = sum(L.U) * sum(L.V)

function Base.size(S::LowRank, k::Int)
    k == 1 ? size(S.U, 1) : size(S.V, k)
end
Base.size(S::LowRank) = (size(S.U, 1), size(S.V, 2))
LinearAlgebra.rank(S::LowRank) = size(S.U, 2)
LinearAlgebra.issuccess(S::LowRank) = S.info ≥ 0
Base.Matrix(S::LowRank) = S.U*S.V

import LinearAlgebra: dot, *, \, /, adjoint
# adjoint(S::LowRank) = Adjoint(S)
# TODO move this to ==, instead of ===, compare speed in case pointers are equal
# WARNING this is actually not completely correct:
# u*v' + v*u' is symmetric and low rank, but
# not of the form that we are checking for here
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

# compute diagonal from low-rank factorization without extraneous operations
function LinearAlgebra.diag(L::LowRank{T}) where {T<:Number}
    n = checksquare(L)
    d = zeros(T, n)
    for i = 1:n
        d[i] += @views dot(L.U[i,:], L.V[:,i])
    end
    d
end

# TODO: take care of multiplication with adjoints of LowRank!
Base.:*(L::LowRank, x::AbstractVector) = L.U*(L.V*x)
Base.:*(x::AbstractVector, L::LowRank) = (x*L.U)*L.V
Base.:*(L::LowRank, A::AbstractMatOrFac) = LowRank(L.U, L.V*A)
Base.:*(A::AbstractMatOrFac, L::LowRank) = LowRank(A*L.U, L.V)

# interaction with Inverse
Base.:*(Inv::Inverse, L::LowRank) = LowRank(Inv * L.U, L.V)
Base.:*(L::LowRank, Inv::Inverse) = LowRank(L.U, L.V * Inv)
Base.:*(Inv::PseudoInverse, L::LowRank) = LowRank(Inv * L.U, L.V)
Base.:*(L::LowRank, Inv::PseudoInverse) = LowRank(L.U, L.V * Inv)

function Base.:*(A::LowRank, B::LowRank)
    C = A.V * B.U
    size(C, 1) ≥ size(C, 2) ? LowRank(A.U*C, B.V) : LowRank(A.U, C*B.V)
end
Base.:\(A::AbstractMatOrFac, L::LowRank) = LowRank(A \ L.U, L.V)
Base.:/(L::LowRank, A::AbstractMatOrFac) = LowRank(L.U, L.V / A)
# least squares solve
# TODO: make possible for matrix
# TODO: do we have to right multiply by V's pinverse?
# function LazyInverse.pseudoinverse(L::LowRank, side::Union{Val{:L}, Val{:R}})
#     LowRank(pinverse(L.V), pinverse(L.U))
# end
# Base.:\(L::LowRank, b::AbstractVector) = pinverse(L) * b #pinverse(L.V) * (pinverse(L.U) * b)
# Base.:/(b::AbstractVector, L::LowRank) = b * pinverse(L) # pinverse(L.V, :R)) * pinverse(L.U, :R)

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
################# algorithms to compute low rank approximation #################
# TODO: lowrank!(::LowRank, typeof(als)) ...
function lowrank end
function als end # alternating least squares
# low rank approximation via als
function lowrank(::typeof(als), A::AbstractMatrix{T}, rank::Int;
                    tol::Real = 1e-12, maxiter::Int = 32) where {T}
    U = rand(eltype(A), (size(A, 1), rank))
    V = rand(eltype(A), rank, size(A, 2))
    U, V, info = als!(U, V, A, tol, maxiter)
    LowRank(U, V; tol = tol, info = info)
end

# uses pivoted cholesky to compute a low-rank approximation to A with tolerance tol
function lowrank(::typeof(cholesky), A::AbstractMatrix, rank::Int = checksquare(A);
                        tol::Real = 1e-12, check::Bool = true)
    cholesky(A, Val(true), Val(false), rank, tol = tol, check = check)
end

function als(A::AbstractMatrix, rank::Int; tol = 1e-12, maxiter = 32)
    n, m = size(A)
    als!(randn(n, rank), randn(rank, m), A, tol, maxiter)
end

# alternating least squares for low rank decomposition
function als!(U::AbstractMatrix, V::AbstractMatrix, A::AbstractMatrix,
                tol::Real = 1e-12, maxiter::Int = 32)
    pals!(U, V, A, tol = tol, maxiter = maxiter)
end

function als!(L::LowRank, A::AbstractMatrix, maxiter = 32)
    als!(L.U, L.V, A, L.tol, maxiter)
end

##################### projected alternating least squares ######################
# updates U, V
function pals!(U::AbstractMatrix, V::AbstractMatrix, A::AbstractMatrix,
                project_u! = identity, project_v! = identity;
                maxiter::Int = 32, tol = eps(eltype(A)))
    info = -1
    project_u!(U)
    project_v!(V) # need to project rows of V
    for i in 1:maxiter
        U .= A / V
        project_u!(U)
        V .= U \ A
        project_v!(V) # need to project rows of V
        if norm(A-U*V) < tol # should we take norm or maximum?
            info = 0
            break
        end
    end
    U, V, info
end

function pals(A::AbstractMatrix, k::Int,
                project_u! = identity, project_v! = identity;
                maxiter::Int = 32, tol = eps(eltype(A)))
    n, m = size(A)
    U = rand(n, k)
    V = rand(k, m)
    pals!(U, V, A, project_u!, project_v!, maxiter = maxiter, tol = tol)
end

function pals!(L::LowRank, A::AbstractMatrix,
                project_u! = identity, project_v! = identity;
                maxiter::Int = 32, tol = eps(eltype(A)))
    pals!(L.U, L.V, A, project_u!, project_v!, maxiter = maxiter, tol = tol)
end

# positive!(x) = (@. x = max(x, 0))

# PU = Projection(U)
# PV = Projection(V')
# function pu!(x)
#     mul!(x, PU, x)
# end
# function pv!(x)
#     x = x'
#     mul!(x, PV, x)
# end
