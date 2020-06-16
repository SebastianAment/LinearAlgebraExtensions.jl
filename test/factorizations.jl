module TestFactorizations
using Test
using LinearAlgebraExtensions: Projection
using LinearAlgebra
@testset "projection" begin
    n, m = 16, 3
    A = randn(n, m)
    P = Projection(A)
    x = randn(n)
    @test P isa Projection
    Px = P * x
    @test P(x) == P*x
    @test Px ≈ P*Px # idempotence
    mul!(Px, P, x)
    @test Px ≈ P*x
    @test P^2 == P
    @test size(P) == (n, n)

    # projecting columns of matrix
    k = 2
    X = randn(n, k)
    PX = P*X
    @test size(PX) == size(X)
    mul!(PX, P, X)
    @test PX ≈ P(PX)

    MP = Matrix(P)
    @test MP^2 ≈ MP

    # with pre-allocation
    T = zeros(size(P.A, 2), k)
    mul!(PX, P, X, T)
    @test PX ≈ P(PX)

    # linearity
    # B = randn(n, m)
    # Q = Projection(B)
end

# TODO: Hadamard

end # TestFactorizations
