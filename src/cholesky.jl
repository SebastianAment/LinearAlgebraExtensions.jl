############# implementation of GENERIC cholesky factorizations ################
# why?
# - works with AbstractMatrix type, which we exploit to not have to construct kernel matrices
# - works for low-rank approximation,
# - the above two combined lead to computational savings because of the sparse access pattern
# - in contrast, the LinearAlgebra implementation only works on StridedMatrices
# -> could add generic pivoted cholesky to LinearAlgebra -> make work for complex hermitian matrices
# potentially problematic assumption: LinearAlgebra assumes that setindex! is implemented
# IDEA: have special factorization for Toeplitz matrix WIP
using LinearAlgebra: Cholesky, CholeskyPivoted
import LinearAlgebra: cholesky, cholesky!, adjoint, dot
# (I+K) = (I+U'U)
# (I+U)'(I+U) = I + K + U' + U
############################### Cholesky #######################################
# non-pivoted cholesky, stores
function cholesky(A::AbstractMatrix, pivoted::Val{false} = Val(false); check::Bool = true)
    U = zeros(eltype(A), size(A))
    info = cholesky!(U, A, Val(false); check = check)
    uplo = 'U'
    return Cholesky{eltype(U),typeof(U)}(U, uplo, info)
end

# also works if U = A (overwrites upper triangular part)
function cholesky!(U::AbstractMatrix, A::AbstractMatrix,
                            pivoted::Val{false} = Val(false); check::Bool = true)
    n = LinearAlgebra.checksquare(A)
    nu = LinearAlgebra.checksquare(U)
    n == nu || error("target matrix does not have the same shape as input matrix")

    d = diag(A)
    info = 0
    @inbounds for i in 1:n
        if d[i] < 0 # negative pivot
            check && throw(LinearAlgebra.PosDefException(i))
            info = -1
            break
        end
        U[i, i] = sqrt(d[i])

        for j = i+1:n # this loop has zero memory allocation
            dot_mj = zero(eltype(A)) # dot product
            @simd for k = 1:i-1
                dot_mj += U[k, i] * U[k, j]
            end
            U[i, j] = (A[i, j] - dot_mj)  / U[i, i]
            d[j] -= U[i, j]^2
        end
    end
    return info
end

############################### Pivoted Cholesky ###############################
# returns PivotedCholesky
function cholesky(A::AbstractMatrix, pivoted::Val{true}, rank::Int = size(A, 1);
                                tol::Real = eps(eltype(A)), check::Bool = true)
    n = LinearAlgebra.checksquare(A)
    U = zeros(eltype(A), (rank, size(A, 2)))
    return cholesky!(U, A, Val(true), rank; tol = tol, check = check)
end

# returns pivots piv, rank m, trace norm bound ε, info (0 if successful, -1 if not symmetric, p.s.d., 1 if low rank)
# triangular controls if we want to return a triangular factorization with pivutation matrices,
# or a generic low rank matrix which alrady incorporates the pivutations
function cholesky!(U::AbstractMatrix, A::AbstractMatrix, pivoted::Val{true},
                rank::Int = size(A, 1); tol::Real = eps(eltype(A)), check::Bool = true)
    size(U, 2) == size(A, 2) || error("input matrix U does not have the same outer dimension as A")
    rank = min(rank, size(U, 1)) # maximum rank can't exceed inner dimension of U
    U .= 0
    return _chol!(U, A, rank, tol, check)
end

# pivoted cholesky which computes upper-triangular U
# returns U s.t. A[piv,piv] = U'U
# reordering d according to pivots, could allow for simd summation for error bound
function _chol!(U::AbstractMatrix, A::AbstractMatrix, rank::Int,
                                tol::Real = eps(eltype(A)), check::Bool = true)
    n = LinearAlgebra.checksquare(A)
    size(U, 1) ≥ rank || error("input matrix U does not have more than rank = $rank rows")
    size(U, 2) == n || error("input matrix U does not have the same outer dimension as A")

    d = diag(A)
    piv = collect(1:n) # pivot indices
    ε = sum(abs, d)
    m = 1
    info = 1
    @inbounds while true
        # find pivot element
        max_d = zero(eltype(A))
        i = Int(m)
        for k in m:n
            if d[piv[k]] > max_d
                max_d = d[piv[k]]
                i = k
            end
        end
        # i ≥ m
        if d[piv[i]] < 0 # negative pivot
            m -= 1
            info = -1
            break
        end
        piv[i], piv[m] = piv[m], piv[i]

        # swap pivoted column in U
        for k in 1:size(U, 1)
            U[k, i], U[k, m] = U[k, m], U[k, i]
        end
        U[m, m] = sqrt(d[piv[m]])

        for j in m+1:n # this loop has zero memory allocation!
            dot_mj = zero(eltype(A)) # dot product
            @simd for k in 1:m-1
                dot_mj += U[k, m] * U[k, j]
            end
            U[m, j] = (A[piv[m], piv[j]] - dot_mj)  / U[m, m]
            d[piv[j]] -= U[m, j]^2
        end

        # calculate trace norm error
        ε = zero(eltype(A))
        for k in m+1:n
            ε += abs(d[piv[k]])
        end

        # termination criterion
        if ε ≤ tol
            break
        elseif rank == m # this branch means ε > tol
            info = -2
            break
        end
        m += 1
    end
    if check && info == -1
        throw(LinearAlgebra.PosDefException(m))
    elseif check && info == -2
        throw("pivoted cholesky algorithm was not able to decrease the " *
        "approximation error ε = $ε below tol = $tol with max_iter = $max_iter steps")
    end
    if m == n # full rank
        info = 0
    end
    uplo = 'U' # denotes that upper triangular part of A was used to calculate the factorization
    return CholeskyPivoted{eltype(U),typeof(U)}(U, uplo, piv, m, ε, info)
end

# computes diagonal of matrix from low rank cholesky factorization
function LinearAlgebra.diag(F::CholeskyPivoted)
    n = size(F.L)[1]
    ip = invpiv(F.p)
    U = @view F.U[1:F.rank, ip]
    d = zeros(eltype(F), n)
    for i = 1:n
        for j = 1:F.rank
            d[i] += U[j,i]^2
        end
    end
    return d
end
