function mutate_weights!(state::State, genotype::Network, mr::Float32; n=-1)
    gene_counter = get_counter(AbstractGene, state)
    weights = get_weights(state.rng, genotype, n=n)
    for weight in weights
        push!(weight.muts, NetworkGene(state.rng, gene_counter, mr))
    end
end

function mutate(state::State, genotype::Network; mr::Float32, n::Int=2)
    @assert genotype.coupling in (StrictCoupling, NoCoupling) "Only strict and no coupling are supported right now"
    rng = state.rng
    new_genotype = deepcopy(genotype)
    genotype.coupling == StrictCoupling && (n = -1)
    mutate_weights!(state, new_genotype, mr, n=n)
    new_genotype
end


