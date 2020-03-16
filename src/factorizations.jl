
########################### Projection Matrix ##################################
# stands for A*(AF\y) = A*inverse(A'A)*(A'y) = A*pseudoinverse(A)
struct Projection{T, AT<:AbstractMatOrFac{T},
                                AFT<:AbstractMatOrFac{T}} <: Factorization{T}
    A::AT
    A⁺::AFT
    # temp::V             V<:AbstractVecOrMat
end
Projection(A::AbstractMatOrFac) = Projection(A, pseudoinverse(qr(A, Val(true)))) # defaults to pivoted qr

# TODO: potentially do memory pre-allocation (including temporary)
(P::Projection)(x::AbstractVecOrMatOrFac) = P.A * (P.A⁺ * x)

Base.size(P::Projection, k::Integer) = 0 < k ≤ 2 ? size(P.A, 1) : 1
Base.size(P::Projection) = (size(P, 1), size(P, 2))

# properties
LinearAlgebra.Matrix(P::Projection) = Matrix(P.A * P.A⁺)
LinearAlgebra.adjoint(P::Projection) = P
Base.:^(P::Projection, n::Integer) = P
function Base.literal_pow(::typeof(^), P::Projection, ::Val{N}) where N
    N > 0 ? P : error("Projection P is not invertible")
end
Base.:*(P::Projection, x::AbstractVecOrMat) = P(x)
Base.:*(x::AbstractVecOrMat, P::Projection) = P(x')'

############################## Hadamard Product ################################
# TODO: tests and non-allocating versions
# TODO: move this into KroneckerProducts?
struct HadamardProduct{T, A<:Tuple{Vararg{AbstractMatOrFac}}} <: Factorization{T}
    factors::A
    function HadamardProduct(A::Tuple{Vararg{AbstractMatOrFac}})
        T = promote_type(eltype.(A)...)
        all(==(size(A[1]), size.(A))) || error("matrices have to have same size to form Hadamard product")
        HadamardProduct{T, typeof(A)}(A)
    end
end
const SchurProduct = HadamardProduct
# smart constructor
hadamard(A::AbstractMatOrFac...) = HadamardProduct(A)
hadamard(H::HadamardProduct, A::AbstractMatOrFac...) = hadamard(tuple(H.args..., A...))
hadamard(A::AbstractMatOrFac, H::HadamardProduct) = hadamard(tuple(A, H.args...))
const ⊙ = hadamard # \odot

Base.size(H::HadamardProduct) = size(H.factors[1])
Base.getindex(H::HadamardProduct, i, j) = prod(A->A[i, j], H.factors)
Base.eltype(H::HadamardProduct{T}) where {T} = T
function Base.Matrix(H::HadamardProduct)
    [H[i,j] for i in 1:size(H, 1), j in 1:size(H, 2)]
end # could also define AbstractMatrix which could preserve sparsity patterns

# from Horn, Roger A.; Johnson, Charles R. (2012). Matrix analysis. Cambridge University Press.
function Base.:*(H::HadamardProduct, x::AbstractVector)
    checksquare(H) # have to check identity for non-square matrices
    x = Diagonal(x)
    for A in H.factors
        x = (A*x)'
    end
    diag(x)
end
# TODO: check this
LazyInverse.inverse(H::HadamardProduct) = hadamard(inverse.(H.factors))
LinearAlgebra.isposdef(H::HadamardProduct) = all(isposdef, H.factors) # this fact makes product kernels p.s.d. (p. 478)
LinearAlgebra.issymmetric(H::HadamardProduct) = all(issymmetric, H.factors)
LinearAlgebra.ishermitian(H::HadamardProduct) = all(ishermitian, H.factors)
LinearAlgebra.issuccess(H::HadamardProduct) = all(issuccess, H.factors)
# TODO: factorize

# function LinearAlgebra.dot(x::AbstractVector, H::Hadamard, y::AbstractVector)
#     tr(Diagonal(x)*H.A*Diagonal(y)*H.B')
# end

############################ Symmetric Rescaling ###############################
# applications for Schur complement, and VerticalRescaling, SoR
# represents D'*A*D
# could be SymmetricRescaling
struct SymmetricRescaling{T, M<:AbstractMatOrFac{T},
                                    N<:AbstractMatOrFac{T}} <: Factorization{T}
    D::M
    A::N
end
import LinearAlgebra: Matrix, *, \, dot, size
Matrix(L::SymmetricRescaling) = L.D' * L.A * L.D

# doing this:
# applied(*, L.D', L.A, L.D)
# wouldn't be the whole story, because we want to define matrix multiplication
# with SymmetricRescaling, leading to efficiency gains with structure

# avoids forming the full matrix
*(L::SymmetricRescaling, B::AbstractVecOrMat) = L.D'*(L.A*(L.D*B))
\(L::SymmetricRescaling, B::AbstractVecOrMat) = L.D'\(L.A\(L.D\B))

function dot(x::T, L::SymmetricRescaling, y::T) where {T<:AbstractArray}
    dot(L.D*x, L.A, L.D*y)
    # memory allocation can be avoided with lazy arrays:
    # Dx = applied(*, L.D, x)
    # Dy = applied(*, L.D, y)
    # dot(Dx, L.A, Dy)
end

# this might be too general to be efficient for the SymmetricRescaling usecase
# struct LazyMatrixProduct{T, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
#     A::M
#     B::M
# end

################
# import LinearAlgebra: +
# function +(S::Symmetric, D::Diagonal)
#     T = copy(S)
#     @inbounds for i in 1:size(D, 1)
#         T.data[i, i] += D[i]
#     end
#     return T
# end
