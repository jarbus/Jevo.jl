export UniformReproducer
@define_op "UniformReproducer" "AbstractReproducer"
UniformReproducer(pop_size::Int, ids::Vector{String}=String[]; kwargs...) =
    create_op("UniformReproducer",
          retriever=PopulationRetriever(ids),
          updater=map((s,p)->uniform_reproduce!(s,p,pop_size))
          ;kwargs...)

function uniform_reproduce!(state::AbstractState, pops::Vector{Population}, size::Int)
    @assert length(pops) == 1 "UniformReproducer only works with a single population"
    @assert 0 < length(pops[1].individuals) < size "Population must have individuals to reproduce and fewer individuals than $size to reproduce uniformly, $(pops[1].id) has $(length(pops[1].individuals)) individuals."
    pop = pops[1]
    n_parents = length(pop.individuals)

    # initialize rest of individuals as undefined
    IndType = typeof(pop.individuals[1])
    append!(pop.individuals, Vector{IndType}(undef, size-n_parents))
    # choose a random parent and push to individuals
    # TODO use rng seed for each child
    rngs = [rand(state.rng, UInt) for i in (n_parents+1):size]
    Threads.@threads for i in (n_parents+1):size
        rng = StableRNG(rngs[i-n_parents])
        parent = pop.individuals[rand(rng, 1:n_parents)]
        child = clone(state, parent)
        pop.individuals[i] = child
    end
    @assert length(pop.individuals) == size "Failed to reproduce $(pop.id) uniformly"
end
