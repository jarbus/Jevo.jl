function mutate_coupled!(state::State, genotype::Network, mr::Float16)
    counter = get_counter(AbstractGene, state)
end

function mutate_uncoupled!(state::State, genotype::Network, mr::Float16)
    counter = get_counter(AbstractGene, state)
end

function mutate(state::State, genotype::Network; mr::Float16)
    # return a new network with added gene
    rng = state.rng
    # create a copy of the network
    new_genotype = deepcopy(genotype)
    if genotype.coupling == StrictCoupling
        mutate_coupled!(state, new_genotype, mr)
    elseif genotype.coupling == NoCoupling
        mutate_uncoupled!(state, new_genotype, mr)
    else
        @error "Coupling scheme not implemented"
    end
    new_genotype
end


