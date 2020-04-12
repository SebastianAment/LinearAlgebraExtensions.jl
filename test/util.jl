module TestUtil

using Test
using LinearAlgebra
using LinearAlgebraExtensions: vector², matrix

@testset "vec²" begin
    m, n = 3, 16
    A = randn(m, n)
    v = vector²(A)
    B = matrix(v)
    for i in 1:n
        @test v[i] ≈ A[:, i]
    end
    @test A ≈ B
end

end
