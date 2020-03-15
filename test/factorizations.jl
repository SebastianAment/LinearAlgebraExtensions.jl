module TestFactorizations
using Test
using LinearAlgebra
n = 3
u = randn(n)
v = randn(n)

@testset "LowRank" begin
    using LinearAlgebraExtensions: LowRank
    A = LowRank(u, v')
    B = LowRank(randn(n), randn(n)')
    MA, MB = Matrix.((A, B))
    @test A + B isa LowRank
    @test A + A' isa LowRank
    @test Matrix(A + B) ≈ MA + MB
    @test Matrix(A + A') ≈ MA + MA'
    @test sum(A) ≈ sum(Matrix(A))
    @test Matrix(LowRank(u)) ≈ u*u'
end

end # TestFactorizations
