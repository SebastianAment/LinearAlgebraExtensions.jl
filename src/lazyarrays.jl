using LinearAlgebra
using LinearAlgebra: checksquare

import Base: getindex, size
import LinearAlgebra: \, /, *, inv, factorize, dot, det

##################### Lazy Multi-Dimensional Grid ##############################
# useful to automatically detect Kronecker structure in Kernel matrices at compile time
struct LazyGrid{T, V<:Tuple{Vararg{AbstractVector{T}}}} <: AbstractVector{AbstractVector{T}}
    args::V
end
Base.length(G::LazyGrid) = prod(length, G.args)
Base.size(G::LazyGrid) = (length(G),)
Base.ndims(G::LazyGrid) = length(G.args) # mh, maybe don't do this?
function Base.getindex(G::LazyGrid{T}, i::Integer) where {T}
    @boundscheck checkbounds(G, i)
    val = zeros(T, ndims(G))
    n = length(G)
    @inbounds for (j, a) in enumerate(G.args)
        n ÷= length(a)
        val[j] = a[cld(i, n)]  # can this be replaced with fld1, mod1?
        i = mod1(i, n) # or some kind of shifted remainder?
    end
    return val
end
grid(args::Tuple{Vararg{AbstractVector}}) = LazyGrid(args)
grid(args::AbstractVector...) = LazyGrid(args)

######################### Lazy Difference Vector ###############################
# TODO: is it worth using the LazyArrays package?
# could be replaced by LazyVector(applied(-, x, y)), which is neat.
# lazy difference between two vectors, has no memory footprint
struct LazyDifference{T, U<:AbstractVecOrTup, V<:AbstractVecOrTup} <: AbstractVector{T} # U, V} <: AbstractVector{T} #
    x::U
    y::V
    function LazyDifference(x, y)
        length(x) == length(y) || throw(DimensionMismatch("x and y do not have the same length: $(length(x)) and $(length(y))."))
        T = promote_type(eltype(x), eltype(y))
        new{T, typeof(x), typeof(y)}(x, y)
    end
end

difference(x::Number, y::Number) = x-y # avoid laziness for scalars
difference(x, y) = LazyDifference(x, y)

size(d::LazyDifference) = (length(d.x),)
getindex(d::LazyDifference, i::Integer) = d.x[i]-d.y[i]
# getindex(d::LazyDifference, ::Colon) = d.x-d.y

# getindex(d::LazyDifference, i::Int) = LazyDifference(x[i], y[i]) # recursive
minus(A::AbstractArray) = ApplyArray(applied(-, A))
# const Minus{T, N} = LazyArray{T, N, typeof(-), ::Tuple{<:AbstractArray{T, N}}}
# const Minus{M<:AbstractArray{T, N}} where {T, N} = ApplyArray{T, N, typeof(-), ::Tuple{M}}

# Minus(A::AbstractArray{T, N}) where {T, N} = Minus{T, N}(A)

############################## PeriodicVector ##################################
using InfiniteArrays
# can use this to create an infinite stream of data
struct PeriodicVector{T, V<:AbstractVector{T}} <: AbstractVector{T}
    x::V
end
Base.getindex(P::PeriodicVector, i) = @inbounds P.x[mod1.(i, length(P.x))]
Base.length(::PeriodicVector) = ∞ # TODO: look at InfiniteArrays.jl for how to do this well
Base.size(::PeriodicVector) = (∞,)
Base.firstindex(::PeriodicVector) = -∞
Base.lastindex(::PeriodicVector) = ∞

# maybe not a good idea?
# Base.setindex!(P::PeriodicVector, v, i) = (P.x[mod1(i, length(P.x))] = v)
# TODO: fft? leads to fft(x) padded with infinitely many zeros on each side

##################### Lazy Matrix Sum and Product ##############################
using LazyArrays
function lazysum(A::Union{AbstractMatrix, UniformScaling}...)
    ApplyMatrix(applied(+, A...))
end
lazyprod(A::AbstractMatrix...) = ApplyMatrix(applied(*, A...))

struct LazySum{T, M} <: AbstractMatrix{T}
    args::M
end
function LazySum(args::Tuple{Vararg{Union{AbstractMatrix, UniformScaling}}})
    T = promote_type((eltype(a) for a in args)...)
    LazySum{T, typeof(args)}(args)
end
Base.size(A::LazySum) = size(A.args[1])
Base.getindex(A::LazySum, i, j) = sum(B[i,j] for B in A.args)

struct LazyProduct{T, M} <: AbstractMatrix{T}
    args::M
end
function LazyProduct(args::Tuple{Vararg{Union{AbstractMatrix, UniformScaling}}})
    T = promote_type((eltype(a) for a in args)...)
    LazyProduct{T, typeof(args)}(args)
end
Base.size(A::LazyProduct) = size(A.args[1])
Base.getindex(A::LazyProduct, i, j) = *(A.args[1][i,:], A.args[2:end-1]..., A.args[end][:,j])

# TODO: optimal ordering algorithm
# LinearAlgebra.Matrix(A::LazyProduct) = *(A.args...)

# Matrix multiplication with optimal ordering
# function LinearAlgebra.:*(A::AbstractMatrix...)
#     #, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix...)
#     cost(A) = 0
#     cost(A, B) = prod(size(A)) * size(B, 2) + cost(A) + cost(B)
#     function cost(A...)
#         k = minimum(cost(A[1:i], A[i+1:end]) for i in 1:length(A)-1)
#     end
# end

#################### Unitary Discrete Fourier Transform ########################
using FFTW
struct ℱ{T} <: AbstractMatrix{T}
    n::Int
end
ℱ(n::Int) = ℱ{Complex{Float64}}(n)

Base.size(F::ℱ) = (F.n, F.n)
Base.getindex(F::ℱ, i, j) = begin println("hre") ; exp(-2π*1im*((i-1)*(j-1))/F.n) / sqrt(F.n)end
function Base.:*(F::ℱ, A::AbstractVector)
    B = fft(A)
    B ./ sqrt(F.n)
end
function Base.:*(F::ℱ, A::AbstractMatrix)
    B = fft(A)
    B ./ sqrt(F.n)
end
function Base.:*(F::Adjoint{<:Number, <:ℱ}, A::AbstractVector)
    B = ifft(A)
    B .*= sqrt(F.parent.n)
end
function Base.:*(F::Adjoint{<:Number, <:ℱ}, A::AbstractMatrix)
    B = ifft(A)
    B .*= sqrt(F.parent.n)
end
Base.:*(A::AbstractVector, F::ℱ) = (F'*A')' # TODO: check dimensions
Base.:*(A::AbstractMatrix, F::ℱ) = (F'*A')' # TODO: check dimensions
Base.:\(F::ℱ, A::AbstractVecOrMat) = F'*A # are these necessary?
Base.:/(A::AbstractVecOrMat, F::ℱ) = (F*A')'
Base.inv(F::ℱ) = F' # TODO: make sure adjoints are lazy

LazyInverse.inverse(F::ℱ) = F'

# first dimension varies most quickly
# function Base.getindex(G::LazyGrid{T}, i::Integer) where {T}
#     @boundscheck checkbounds(G, i)
#     val = zeros(T, ndims(G))
#     n = length(G)
#     j = ndims(G)
#     @inbounds for a in reverse(G.args)
#         n ÷= length(a)
#         val[j] = a[cld(i, n)]  # can this be replaced with fld1, mod1?
#         i = mod1(i, n) # or some kind of shifted remainder?
#         j -= 1
#     end
#     return val
# end
