module LinearAlgebraExtensions
using LinearAlgebra
using LinearAlgebra: checksquare
using LazyInverse
using LazyInverse: inverse, Inverse, pinverse, pseudoinverse, PseudoInverse

include("util.jl")
include("extensions.jl")
include("lazyarrays.jl")
include("factorizations.jl")
include("lowrank.jl")
include("cholesky.jl")
include("random.jl")

end
