module TestLazyArrays

using LinearAlgebra
using Test

@testset "difference" begin
    using LinearAlgebraExtensions: difference, LazyDifference
    n = 128
    x = randn(n)
    y = randn(n)
    d = difference(x, y)
    @test d isa LazyDifference
    @test d == x-y

    x = randn()
    y = 1.
    d = difference(x, y)
    @test d isa Number
    @test d == x-y

    x, y = randn(1), randn(1)
    d = difference(x, y)
    @test d isa AbstractVector
    @test d[1] == (x-y)[1]
    @test length(d) == 1
end

@testset "Fourier" begin
    using LinearAlgebraExtensions: ℱ
    using FFTW
    n = 128
    x = randn(n)
    F = ℱ(n)
    Fx = F*x
    @test Fx == fft(x) / √n
    @test real.(F'*Fx) ≈ x
    @test real.(F\Fx) ≈ x
    @test real.(x*F/F) ≈ x
    @test inv(F) isa Adjoint{<:Number, <:ℱ}
end

@testset "LazyGrid" begin
    using LinearAlgebraExtensions: LazyGrid, grid
    n = 8
    x = randn(n)
    m = 6
    y = range(1, m, length = m)
    g = grid(x, x, y)
    @test g isa LazyGrid
    @test eltype(g) == Vector{Float64}
    @test length(g) == length(x)^2 * length(y)
    @test ndims(g) == 3

    # test correct indexing (equivalent to column major order)
    @test g[2] ≈ [x[2], x[1], y[1]]
    @test g[n+1] ≈ [x[1], x[2], y[1]]
    @test g[2n+1] ≈ [x[1], x[3], y[1]]
    @test g[n^2+1] ≈ [x[1], x[1], y[2]]

    # test row major indexing
    @test getindex(g, 2, Val(true)) ≈ [x[1], x[1], y[2]]
    @test getindex(g, m+1, Val(true)) ≈ [x[1], x[2], y[1]]
    @test getindex(g, n*m+1, Val(true)) ≈ [x[2], x[1], y[1]]

    # combine arrays of different dimensionality
    x = randn(n)
    y = randn(2, n)
    g = grid(x, y)
    @test g isa LazyGrid
    @test eltype(g) == Vector{Float64}
    @test length(g) == length(x) * length(y)
    @test ndims(g) == 3
    @test g[1] == [x[1], y[:,1]...]
end
end # TestLazyArrays
