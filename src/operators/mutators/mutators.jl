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
    Threads.@threads for ind in pop.individuals
        if ind.generation == gen
            mutate!(state, ind; kwargs...)
        end
    end
end
function mutate!(state::AbstractState, ind::AbstractIndividual; kwargs...)
    ind.genotype = mutate(state, ind.genotype; kwargs...)
end

mutate(::AbstractState, genotype::AbstractGenotype; kwargs...) =
    error("mutate function not implemented for $(typeof(genotype))")
