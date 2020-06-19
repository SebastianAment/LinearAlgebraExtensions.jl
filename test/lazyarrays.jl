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
    y = 1:1.0:6
    g = grid(x, x, y)
    @test g isa LazyGrid
    @test eltype(g) == Float64
    @test length(g) == length(x)^2 * length(y)
    @test ndims(g) == 3

end
end # TestLazyArrays
