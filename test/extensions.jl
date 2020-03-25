module TestExtensions
using Test
using LinearAlgebra
using LinearAlgebraExtensions

@testset "logabsdet" begin
    n = 16
    D = Diagonal(randn(n))
    d, s = logabsdet(D)
    @test s*exp(d) ≈ det(D)
end

end
