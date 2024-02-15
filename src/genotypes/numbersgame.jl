export VectorGenotype
mutable struct VectorGenotype <: AbstractGenotype
    numbers::Vector{Float32}
end

VectorGenotype(n::Int, rng::AbstractRNG; init::Function=rand) = VectorGenotype(init(rng, Float32, n))
