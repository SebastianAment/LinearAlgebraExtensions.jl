module TestLowRank
using Test
using LinearAlgebra
using LazyInverse: inverse

@testset "LowRank" begin
    using LinearAlgebraExtensions: LowRank
    n = 3
    u = randn(n)
    v = randn(n)

    A = LowRank(u, v')
    B = LowRank(randn(n), randn(n)')
    MA, MB = Matrix.((A, B))
    @test MA ≈ u*v'
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
    @test rank(A) == 1 == rank(MA)
    @test Matrix(R\A) ≈ R\MA
    @test R*A isa LowRank
    @test A*R isa LowRank
    @test Matrix(R*A) ≈ R*MA
    @test A*B isa LowRank
    @test Matrix(A*B) ≈ MA*MB
    @test eltype(A) == Float64

    # trace
    @test tr(A) ≈ tr(MA)
    @test tr(B) ≈ tr(MB)

    # interaction with lazy inverse
    @test Matrix(inverse(R)*A) ≈ Matrix(R\A)
    @test Matrix(A*inverse(R)) ≈ Matrix(A/R)

    # product of low rank matrices preserves lowest rank
    C = LowRank(randn(3,2))
    @test rank(C) == 2
    @test rank(A*C) == 1
    @test Matrix(A*C) ≈ Matrix(A)*Matrix(C)
end

end # TestLowRank
