export VectorGenotype
mutable struct VectorGenotype <: AbstractGenotype
    numbers::Vector{Float32}
end

VectorGenotype(n::Int, rng::AbstractRNG; init::Function=rand) = VectorGenotype(init(rng, Float32, n))

function mutate(state::State, genotype::VectorGenotype)
    # add random noise to two random dimensions
    i = rand(state.rng, 1:length(genotype.numbers))
    j = rand(state.rng, 1:length(genotype.numbers))
    while i == j
        j = rand(state.rng, 1:length(genotype.numbers))
    end
    genotype.numbers[i] += randn(state.rng)
    genotype.numbers[j] += randn(state.rng)
    genotype
end
