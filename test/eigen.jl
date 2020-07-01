module TestEigen
using Test
using LinearAlgebra
using LinearAlgebraExtensions
using LinearAlgebraExtensions: ldiv!!, inverse, inverse!

@testset "eigen" begin
    n = 16
    A = randn(n, n)
    A = Symmetric(A)
    E = eigen(A)
    x = randn(n)
    y = A*x
    @test E\y ≈ x

    x2 = similar(x)
    ldiv!!(x2, E, y) # doesn't allocate
    @test x2 ≈ x # correctness
    @test !(y ≈ A*x) # and overwrites y
    @test E*x ≈ A*x # multiplication
    @test Matrix(inverse!(E)) ≈ inv(A)
    @test Matrix(inverse!(E)) ≈ A
end

end
