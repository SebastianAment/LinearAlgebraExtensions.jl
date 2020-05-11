using LinearAlgebra
using LinearAlgebraExtensions
using LinearAlgebraExtensions: als, pals

# TODO: minimize memory allocation for ICM
# TODO: put this into LinearAlgebraExtensions
# non-negative matrix factorization
struct NMF{T} <: Factorization{T}
    W::Matrix{T}
    H::Matrix{T}
end

NMF{T}(n::Int, k::Int, m::Int) where {T} = NMF{T}(rand(T,(n,k)), rand(T, (k, m)))
NMF(n, k, m) = NMF{Float64}(n, k, m)

# TODO: make non-allocating with mul!

# iterative conditional modes algorithm
# from Schmidt et al "Bayesian non-negative matrix factorization" 2009
function ICM!(nmf::NMF{T}, X::M;
            α = T(1), # exponential rate of prior on entries of W
            β = T(1), # " H
            σ = T(1e-8),
            max_iter::Int = Int(64),
            δ::T = T(1e-8)) where {T<:Real, M<:AbstractMatrix{T}}
    W = nmf.W
    H = nmf.H
    (N, K) = size(W)
    iter = 0
    while true
        C = H*H'
        D = X*H'
        # println(C)
        for k = 1:K
            ind = (1:K .!= k)
            w = (D[:,k] - W[:, ind] * C[ind, k] .- α*σ^2) ./ (C[k, k] + δ)
            W[:,k] .= max.(0, w)
        end
        # println(W)

        E = W'*W
        F = W'*X
        for k = 1:K
            ind = (1:K .!= k)
            h = (F[k,:]' - E[k, ind]' * H[ind, :] .- β*σ^2) ./ (E[k, k] + δ)
            H[k, :] .= max.(0, h[:])
        end

        iter += 1
        if iter > max_iter
            break
        end
    end
    nmf.W .= W
    nmf.H .= H
end

# from Lee & Seung 2001: Algorithms for Non-negative Matrix Factorization
function lee_seung!(nmf::NMF, X::M;
            max_iter = 1024) where {T<:Real, M<:AbstractMatrix{T}}
    iter = 0
    W = nmf.W
    H = nmf.H

    WX = W'*X
    WW = W'W
    WWH = (WW)*H

    XH = X*H'
    HH = H*H'
    WHH = W*HH
    ε = eps(T)
    while true

        mul!(WX, W', X)
        mul!(WW, W', W)
        mul!(WWH, WW, H)
        @. H = H * WX / (WWH + ε)

        mul!(XH, X, H')
        mul!(HH, H, H')
        mul!(WHH, W, HH)
        @. W = W * XH / (WHH + ε)

        iter += 1
        if iter > max_iter
            break
        end
    end
    nmf.W .= W
    nmf.H .= H
end

Base.Matrix(n::NMF) = n.W * n.H
LinearAlgebraExtensions.LowRank(n::NMF) = LowRank(n.W, n.H)

#
# n = 16
# k = 2
# m = 16
# W = max.(randn(n, k), 0)
# H = max.(randn(k, m), 0)
# X = W*H
#
# nmf = NMF(W, H)


# using Test
# tol = 1e-3
# println("ICM")
# nmf2 = NMF(n, k, m)
#
# @time ICM!(nmf2, X);
# @test maximum(Matrix(nmf2) - X) < tol
#
# tol = 1e-2
# println("lee_seung")
# nmf3 = NMF(n, k, m)
# println(norm(Matrix(nmf3)-X))
# @time lee_seung!(nmf3, X, max_iter = 1024)
# println(norm(Matrix(nmf3)-X))
# @test maximum(Matrix(nmf3) - X) < tol
