export CloneUniformReproducer, N_OffspringReproducer
"""
    CloneUniformReproducer(pop_size::Int, ids::Vector{String}=String[]; kwargs...)

Creates an [Operator](@ref) that choose individuals uniformly at random and clones them, creating children with identical DNA, until the population reaches `pop_size` individuals.

Should be used after a selector.

See also: [TruncationSelector](@ref)
"""
@define_op "CloneUniformReproducer" "AbstractReproducer"
CloneUniformReproducer(pop_size::Int, ids::Vector{String}=String[]; kwargs...) =
    create_op("CloneUniformReproducer",
          retriever=PopulationRetriever(ids),
          updater=map((s,p)->uniform_reproduce!(s,p,pop_size, clone))
          ;kwargs...)

function uniform_reproduce!(state::AbstractState, pops::Vector{Population}, size::Int, copy_fn::Function)
    @assert length(pops) == 1 "uniform_reproduce only works with a single population"
    @assert 0 < length(pops[1].individuals) < size "Population must have individuals to reproduce and fewer individuals than $size to reproduce uniformly, $(pops[1].id) has $(length(pops[1].individuals)) individuals."
    pop = pops[1]
    n_parents = length(pop.individuals)

    # initialize rest of individuals as undefined
    IndType = typeof(pop.individuals[1])
    append!(pop.individuals, Vector{IndType}(undef, size-n_parents))
    # choose a random parent and push to individuals
    rngs = [rand(state.rng, UInt) for i in (n_parents+1):size]
    Threads.@threads for i in (n_parents+1):size
        rng = StableRNG(rngs[i-n_parents])
        parent = pop.individuals[rand(rng, 1:n_parents)]
        child = copy_fn(state, parent)
        pop.individuals[i] = child
    end
    @assert length(pop.individuals) == size "Failed to reproduce $(pop.id) uniformly"
end

"Makes `n_offspring` copies of each individual in the population."
@define_op "N_OffspringReproducer" "AbstractReproducer"
N_OffspringReproducer(n_offspring::Int, ids::Vector{String}=String[]; max_pop_size=nothing, kwargs...) =
    create_op("N_OffspringReproducer",
          retriever=PopulationRetriever(ids),
          updater=map(map((s,p)->n_offspring_reproduce!(s,p,n_offspring, clone, max_pop_size)))
          ;kwargs...)

function n_offspring_reproduce!(state::AbstractState, pop::Population, n_offspring::Int, copy_fn::Function, max_pop_size=nothing)
    parents = pop.individuals
    n_parents = length(pop.individuals)

    @assert n_parents <= max_pop_size "Population must have fewer individuals than $max_pop_size to reproduce uniformly, $(pop.id) has $(length(pop.individuals)) individuals but max_pop_size is $max_pop_size."

    pop_size = min(n_parents + n_offspring*n_parents, max_pop_size)
    n_children = pop_size - n_parents
    pop.individuals = Vector{typeof(pop.individuals[1])}(undef, pop_size)
    for i in 1:n_parents
        pop.individuals[i] = parents[i]
    end
    # initialize rest of individuals as undefined
    Threads.@threads for i in 1:n_children
        parent_idx = mod1(i, n_parents)
        parent = pop.individuals[parent_idx]
        child = copy_fn(state, parent)
        pop.individuals[n_parents+i] = child
    end
    @info "Population $(pop.id) now has $(length(pop.individuals)) individuals"
    # confirm all individuals are assigned
    @assert all(i-> isassigned(pop.individuals,i), eachindex(pop.individuals)) "Failed to reproduce $(pop.id) uniformly"

end
