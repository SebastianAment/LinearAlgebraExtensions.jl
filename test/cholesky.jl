module TestCholesky

using Test
using LinearAlgebra
using LinearAlgebraExtensions
using LinearAlgebraExtensions: cholesky, cholesky!, LowRank

struct KernelMatrix{T, K, X, Y} <: AbstractMatrix{T}
    k::K
    x::X
    y::Y
    function KernelMatrix(k::K, x::X, y::Y) where {T, K, X<:AbstractVector{T}, Y<:AbstractVector{T}}
        new{T, K, X, Y}
    end
end
Base.size(K::KernelMatrix) = (length(x), length(y))
Base.getindex(K::KernelMatrix, i, j) = K.k(x[i], y[j])

# TODO: more tests

@testset "cholesky" begin
    tol = 1e-12
    k = 16
    n = 32
    A = randn(k, n)
    A = A'A
    C = cholesky(A, Val(true), Val(true); tol = tol, check = false)
    @test issuccess(C)
    @test maximum(Matrix(C)-A) < tol
    @test rank(C) == k

    # testing low rank approximation with non-triangular return factorization
    L = cholesky(A, Val(true), Val(false); tol = tol)
    @test L isa LowRank
    @test rank(L) == k
    @test L.U * L.V ≈ A


    # termination criterion test,
    # this matrix causes tremendous round-off error
    A = [0.18405 0.110636 -0.699126;
    0.110636 0.066505 -0.420257;
    -0.699126 -0.420257 2.65567]
    # even stdlibs pivoted cholesky has problems with it!
    # C = cholesky(A, Val(true), check= false)
    # here, we just test, that if we can't approximate a matrix to
    # within tol, that an error is thrown
    v = false
    try cholesky(A, Val(true), Val(false), 1; tol = 0.) catch; v = true end
    @test v

    v = false
    try cholesky(A, Val(true); tol = 0.) catch; v = true end
    @test v

    # however, if we increase our tolerance, everything works well
    tol = 1e-6
    L = cholesky(A, Val(true), Val(false); tol = tol)
    @test isapprox(Matrix(L), A, atol = tol)
end

end
