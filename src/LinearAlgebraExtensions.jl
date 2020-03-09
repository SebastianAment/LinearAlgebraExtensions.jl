module LinearAlgebraExtensions
using LinearAlgebra
using LazyInverse

include("extensions.jl")
include("lazyarrays.jl")
include("factorizations.jl")
include("cholesky.jl")
include("random.jl")

end