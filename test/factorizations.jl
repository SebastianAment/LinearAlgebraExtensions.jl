module TestFactorizations
using Test
using LinearAlgebra


@testset "LowRank" begin
    using LinearAlgebraExtensions: LowRank
    n = 3
    u = randn(n)
    v = randn(n)

    A = LowRank(u, v')
    B = LowRank(randn(n), randn(n)')
    MA, MB = Matrix.((A, B))
    @test A + B isa LowRank
    @test A + A' isa LowRank
    @test Matrix(A + B) ≈ MA + MB
    @test Matrix(A + A') ≈ MA + MA'

    @test sum(A) ≈ sum(Matrix(A))

    @test Matrix(LowRank(u)) ≈ u*u'
    @test issymmetric(LowRank(u))
    @test ishermitian(LowRank(u))
    @test !ishermitian(A)

    R = randn(n, n)
    @test R\A isa LowRank
    @test Matrix(R\A) ≈ R\MA
    @test R*A isa LowRank
    @test A*R isa LowRank
    @test Matrix(R*A) ≈ R*MA
    @test A*B isa LowRank
    @test Matrix(A*B) ≈ MA*MB
    @test eltype(A) == Float64
    @test tr(A) ≈ tr(MA)

end

end # TestFactorizations
