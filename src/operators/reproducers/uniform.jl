export UniformReproducer
struct UniformReproducer <: AbstractReproducer
    condition::Function
    retriever::AbstractRetriever
    operator::Function
    updater::Function
end
function uniform_reproduce!(state::AbstractState, pops::Vector{Population}, size::Int)
    @assert length(pops) == 1 "UniformReproducer only works with a single population"
    @assert 0 < length(pops[1].individuals) < size "Population must have individuals to reproduce and fewer individuals than $size to reproduce uniformly, $(pops[1].id) has $(length(pops[1].individuals)) individuals."
    pop = pops[1]
    n_parents = length(pop.individuals)
    ind_counter = get_counter(AbstractIndividual, state)
    # choose a random parent and push to individuals
    for _ in (n_parents+1):size
        parent = pop.individuals[rand(1:n_parents)]
        child = clone(state, parent)
        push!(pop.individuals, child)
    end
    @assert length(pop.individuals) == size "Failed to reproduce $(pop.id) uniformly"
end
function UniformReproducer(pop_size::Int, ids::Vector{String}=String[])
    UniformReproducer(always,
                      PopulationRetriever(ids),
                      noop,
                      map((s,p)->uniform_reproduce!(s,p,pop_size)))
end
