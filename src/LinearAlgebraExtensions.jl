module LinearAlgebraExtensions
using LinearAlgebra
using LinearAlgebra: checksquare
using LazyInverse
using LazyInverse: inverse, Inverse, pseudoinverse, PseudoInverse

include("extensions.jl")
include("lazyarrays.jl")
include("factorizations.jl")
include("lowrank.jl")
include("cholesky.jl")
include("random.jl")

end
