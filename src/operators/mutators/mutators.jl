export Mutator

"""
    Mutator(ids::Vector{String}=String[]; time::Bool=false, condition=always, kwargs...)

Operator that mutates the genotypes of individuals in populations with ids in `ids`. Calls [mutate!](@ref) on each individual in each population.
"""
@define_op "Mutator" "AbstractMutator"
Mutator(ids::Vector{String}=String[]; time::Bool=false, condition=always, kwargs...) = 
    create_op("Mutator", 
              condition=condition,
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)->mutate!(s, p; kwargs...))),
              time=time;)

# Mutate all inds made this generation
function mutate!(state::AbstractState, pop::Population, args...; kwargs...)
    gen = generation(state)
    # Prevent using the same RNG twice from different threads
    rngs = [StableRNG(rand(state.rng, UInt)) for i in 1:length(pop.individuals)]
    Threads.@threads for i in 1:length(pop.individuals)
        ind = pop.individuals[i]
        if ind.generation == gen
            mutate!(rngs[i], state, pop, ind, args...; kwargs...)
        end
    end
end

"""
    mutate!(rng::AbstractRNG, state::AbstractState, population::AbstractPopulation, genotype::AbstractGenotype; kwargs...)


"""
function mutate!(rng::AbstractRNG, state::AbstractState, population::AbstractPopulation, ind::AbstractIndividual, args...; fn=mutate, kwargs...)
    ind.genotype = fn(rng, state, population, ind.genotype, args...; kwargs...)
end
