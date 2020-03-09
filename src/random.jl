# randomized linear algebra
using LinearAlgebra
using StatsBase: mean

# TODO: add caluclation of variance
# add second matrix input for gp application
# stochastic trace estimator
# function LinearAlgebra.tr(A::AbstractMatOrFac, B::AbstractMatOrFac; stochastic::Val{true})
#     hutchinson_trace(A)
# end

# rademacher distribution
rademacher(T::DataType = Float64) = rand((T(-1), T(1)))
# function rademacher(T::Type, n::Int)
#    v = Vector{T}(undef, n)
#    @. v = rademacher(T)
# end
# rademacher(n::Int) = rademacher(Float64, n)

# computes stochastic approximation to tr(AB)
function hutchinson_trace(A::AbstractMatOrFac, B::AbstractMatOrFac)
    n = size(A, 1) == size(B, 2) ? size(A, 1) : throw(DimensionMismatch("A*B not square"))
    z = randn(eltype(A), n) # or could be rademacher vector!,
    # za = z; zb = copy(za) # TODO: need mul! for inverse!
    dot(z'A, B*z) # could pre-allocate temporaries for several iterations
end
hutchinson_trace(A::AbstractMatOrFac, B::AbstractMatOrFac, n::Integer) =
                                                mean(_->hutchinson_trace(A), 1:n)
