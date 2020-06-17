module TestFactorizations
using Test
using LinearAlgebraExtensions: Projection, projection
using LinearAlgebra
@testset "projection" begin
    n, m = 16, 4
    A = randn(n, m)
    P = projection(A)

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
    T = zeros(size(P.Q, 2), k)
    mul!(PX, P, X, T)
    @test PX ≈ P(PX)

    # test with rank-deficient A
    r = 2
    A = randn(n, r) * randn(r, m)
    P = projection(A, tol = 1e-10)
    @test size(P.Q, 2) == r

    # test special constructor
    P2 = Projection(P.Q, check = true)
    @test P2.Q ≡ P.Q

    # this throws correclty
    # P2 = Projection(A, compute_projection = false)

    # linearity
    # B = randn(n, m)
    # Q = Projection(B)
end

# TODO: Hadamard

end # TestFactorizations
