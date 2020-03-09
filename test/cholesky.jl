module TestCholesky

using Test
using LinearAlgebraExtensions: cholesky!
tol = 1e-12
k = 16
n = 32
A = randn(k, n)
A = A'A
B = similar(A)
cholesky!(B, A, Val(true); v = Val(true), tol = tol)
@test maximum(B'B-A) < tol

# TODO: more tests
end
