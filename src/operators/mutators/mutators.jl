export Mutator

@define_op "Mutator" "AbstractMutator"
Mutator(ids::Vector{String}=String[]; kwargs...) = 
    create_op("Mutator", 
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)->mutate!(s, p))),
              kwargs...)

# Mutate all inds made this generation
function mutate!(state::AbstractState, pop::Population)
    gen = generation(state)
    for ind in pop.individuals
        if ind.generation == gen
            mutate!(state, ind)
        end
    end
end
function mutate!(state::AbstractState, ind::AbstractIndividual)
    ind.genotype = mutate(state, ind.genotype)
end

mutate(::AbstractState, genotype::AbstractGenotype) =
    error("mutate function not implemented for $(typeof(genotype))")
