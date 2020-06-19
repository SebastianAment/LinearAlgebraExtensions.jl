module TestUtil

using Test
using LinearAlgebra
using LinearAlgebraExtensions: vecofvec, matrix

@testset "vecofvec" begin
    m, n = 3, 16
    A = randn(m, n)
    v = vecofvec(A)
    B = matrix(v)
    for i in 1:n
        @test v[i] ≈ A[:, i]
    end
    @test A ≈ B
end

end
