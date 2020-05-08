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

    # diag
    @test diag(A) ≈ diag(MA)

    # interaction with lazy inverse
    @test Matrix(inverse(R)*A) ≈ Matrix(R\A)
    @test Matrix(A*inverse(R)) ≈ Matrix(A/R)

    # product of low rank matrices preserves lowest rank
    C = LowRank(randn(3,2))
    @test rank(C) == 2
    @test rank(A*C) == 1
    @test Matrix(A*C) ≈ Matrix(A)*Matrix(C)

    # size
    n, k, m = 4, 2, 3
    U, V = randn(n, k), randn(k, m)
    A = LowRank(U, V)
    MA = Matrix(A)
    println(size(A) == size(MA))
    @test size(A) == size(MA)
    @test size(A, 1) == size(MA, 1)
    @test size(A, 2) == size(MA, 2)
    @test size(A, 3) == size(MA, 3)

end

@testset "low-rank algorithms" begin
    using LinearAlgebraExtensions: lowrank, als, als!, pals!
    # low rank approximation algorithms
    n = 61
    m = 17
    k = 8
    U = randn(n, k) / sqrt(n*k) # random matrix with norm ≈ 1
    V = randn(k, m) / sqrt(k*m)
    L = LowRank(U, V)
    A = Matrix(L)
    tol = 1e-12
    ALS = lowrank(als, A, k; tol = tol, maxiter = 32)
    @test issuccess(ALS)
    @test norm(Matrix(ALS)-A) < tol

    L = LowRank(U)
    A = Matrix(L)

    C = lowrank(cholesky, A, 2k; tol = tol) # allowing lee-way in rank determination
    @test issuccess(C)
    @test norm(Matrix(C)-A) < tol
    @test rank(C) == k # check if it found the correct rank

    # projected least squares
    using LinearAlgebraExtensions: Projection

    n = 16
    m = 16
    k = 4
    U = rand(n, k)
    V = rand(k, m)
    A = U*V

    PU = Projection(U)
    function pu!(x)
        mul!(x, PU, x)
    end
    PV = Projection(V')
    function pv!(x)
        x = x'
        mul!(x, PV, x)
    end

    @test PU(U) ≈ U
    @test PV(V')' ≈ V
    CV = copy(V)
    pv!(CV)
    @test V ≈ CV

    L = LowRank(rand, n, k, m)
    pals!(L, A, pu!)
    @test A ≈ Matrix(L)

end

end # TestLowRank
