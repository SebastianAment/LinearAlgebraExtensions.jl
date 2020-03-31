module TestCholesky

using Test
using LinearAlgebra
using LinearAlgebraExtensions
using LinearAlgebraExtensions: cholesky, cholesky!

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
end

# TODO: more tests
end
