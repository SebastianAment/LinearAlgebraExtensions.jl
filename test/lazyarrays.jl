module TestLazyArrays

using LinearAlgebra
using Test

@testset "difference" begin
    using LinearAlgebraExtensions: difference, LazyDifference
    n = 128
    x = randn(n)
    y = randn(n)
    d = difference(x, y)
    @test d == x-y

    x = randn(1)
    y = 1.
    d = difference(x, y)
    @test d isa Number
    @test d[1] == x[1]-y
    d = difference(y, x)
    @test d isa Number
    @test d[1] == y-x[1]
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

end # TestLazyArrays
