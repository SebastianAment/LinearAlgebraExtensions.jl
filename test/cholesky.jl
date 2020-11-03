module TestCholesky

using Test
using LinearAlgebra
using LinearAlgebraExtensions
using LinearAlgebraExtensions: cholesky, cholesky!, LowRank

struct KernelMatrix{T, K, X<:AbstractVector{T}} <: AbstractMatrix{T}
    k::K
    x::X
end
Base.size(K::KernelMatrix) = (length(K.x), length(K.x))
Base.getindex(K::KernelMatrix, i::Int, j::Int) = K.k(K.x[i], K.x[j])
Base.getindex(K::KernelMatrix, i, j) = K.k.(K.x[i], K.x[j]')

@testset "cholesky" begin
    tol = 1e-12
    k = 16
    n = 32
    A = randn(k, n)
    A = A'A
    C = cholesky(A, Val(true), n; tol = tol, check = false)
    @test issuccess(C)
    @test maximum(Matrix(C)-A) < tol
    @test rank(C) == k

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
    try cholesky(A, Val(true), n; tol = 0.) catch; v = true end
    @test v

    # testing with user-defined AbstractMatrix
    k(x, y) = dot(x, y)
    n = 16
    x = randn(n)
    K = KernelMatrix(k, x)
    tol = 1e-8
    C = cholesky(K, Val(true), tol = tol)
    @test issuccess(C)
    @test rank(C) == 1
    @test isapprox(Matrix(C), K, atol = tol)

    h(x) = exp(-x^2/2)
    h(x, y) = h(norm(x-y))
    n = 32
    x = randn(n)
    K = KernelMatrix(h, x)
    C = cholesky(K, Val(true), tol = tol)
    @test issuccess(C)
    @test rank(C) < n
    @test isapprox(Matrix(C), K, atol = tol)
end

end

# code used for benchmarking
# k(x) = exp(-x/2)
# k(x, y) = k(norm(x-y))
# n = 1024
# x = 10randn(n)
# K = KernelMatrix(k, x)
# @time M = Symmetric(Matrix(K))
# println("regular")
# C = cholesky(M)
# @time C = cholesky(M)
# C = cholesky(K)
# @time C = cholesky(K)
# println("pivoted")
# tol = 1e-12
# C = cholesky(M, Val(true), tol = tol, check = false)
# @time C = cholesky(M, Val(true), tol = tol, check = false)
# C = cholesky(K, Val(true), tol = tol)
# @time C = cholesky(K, Val(true), tol = tol)
# println(rank(C))
