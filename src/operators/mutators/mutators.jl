export Mutator

@define_op "Mutator" "AbstractMutator"
Mutator(ids::Vector{String}=String[]; time::Bool=false, kwargs...) = 
    create_op("Mutator", 
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)->mutate!(s, p; kwargs...))),
              time=time;)

# Mutate all inds made this generation
function mutate!(state::AbstractState, pop::Population; kwargs...)
    gen = generation(state)
    # Prevent using the same RNG twice from different threads
    rngs = [StableRNG(rand(state.rng, UInt)) for i in 1:length(pop.individuals)]
    Threads.@threads for i in 1:length(pop.individuals)
        ind = pop.individuals[i]
        if ind.generation == gen
            mutate!(rngs[i], state, ind; kwargs...)
        end
    end
end
function mutate!(rng::AbstractRNG, state::AbstractState, ind::AbstractIndividual; kwargs...)
    ind.genotype = mutate(rng, state, ind.genotype; kwargs...)
end
