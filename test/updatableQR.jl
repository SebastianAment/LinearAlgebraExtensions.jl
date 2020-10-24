module TestUpdatableQR
using Test
using LinearAlgebra
using LinearAlgebraExtensions: UpdatableQR, UQR, remove_column!, add_column!, PUQR

@testset "UpdatableQR" begin
    n, m = 8, 4
    A = randn(n, m)
    uqr = UpdatableQR(A)
    for i in 1:m
        uqr = UpdatableQR(A)
        remove_column!(uqr, i)
        @test A[:, 1:m .!= i] ≈ Matrix(uqr)
        @test uqr.Q * uqr.Q' ≈ I(n) # maintains orthogonality
    end
    uqr = UpdatableQR(A)
    an = randn(n)
    add_column!(uqr, an)
    @test [A an] ≈ Matrix(uqr)
    @test uqr.Q * uqr.Q' ≈ I(n) # maintains orthogonality

    # testing adding column at arbitrary index
    an = randn(n)
    for k in 1:m
        uqr = UpdatableQR(A)
        add_column!(uqr, an, k)
        @test uqr.Q * uqr.Q' ≈ I(n) # maintains orthogonality
        @test [A[:, 1:k-1] an A[:, k:end]] ≈ Matrix(uqr)
    end

    # testing PermutedUpdatableQR
    uqr = UQR(A)
    puqr = PUQR(A)
    @test puqr isa PUQR
    @test size(puqr) == size(A)
    @test size(puqr, 1) == size(A, 1)
    @test size(puqr, 2) == size(A, 2)
    @test eltype(puqr) == eltype(A)
    b = randn(n)
    @test puqr \ b ≈ A \ b

    an = randn(n)
    for k in 1:m # tests adding an removing column at arbitrary index
        # adding column
        add_column!(puqr, an, k)
        add_column!(uqr, an, k)
        @test Matrix(puqr) ≈ Matrix(uqr)

        # solves
        @test puqr \ b ≈ uqr \ b

        # removing column
        remove_column!(puqr, k)
        remove_column!(uqr, k)
        @test Matrix(puqr) ≈ A
    end
end


# benchmark
bench = false
if bench
    n, m = 512, 128
    A = randn(n, m)
    uqr = UpdatableQR(A)
    puqr = UQR(A)

    an = randn(n)
    # k = m ÷ 2
    k = 1
    uqr = UQR(A)

    add_column!(uqr, an)
    remove_column!(uqr)
    @time add_column!(uqr, an, k)
    @time add_column!(uqr, an, k)
    @time remove_column!(uqr)
    @time remove_column!(uqr)

    puqr = PUQR(A)
    add_column!(puqr, an)
    remove_column!(puqr)
    @time add_column!(puqr, an, k)
    @time add_column!(puqr, an, k)
    @time remove_column!(puqr)
    @time remove_column!(puqr)


    # C = copy(uqr)
    # add_column!(C, an, m+1)
    # add_column!(uqr, an)
    # copy is actually most expensive!
    # @btime begin
    #     copy!($C, $uqr)
    #     # add_column!($C, $an, m+1)
    # end

    # @btime begin
    #     copy!($C, $uqr)
    #     remove_column!($C, m)
    # end
end

end # TestUpdatableQR
