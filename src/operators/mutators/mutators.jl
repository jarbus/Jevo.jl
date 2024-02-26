export Mutator
struct Mutator <: AbstractMutator 
    condition::Function 
    retriever::AbstractRetriever # returns vec{vec{ind}} to mutate
    operator::Function  # returns iterable of mutated individuals,
                        # does not update the state
    updater::Function   # adds mutated individuals to the respective 
                               # populations
    data::Vector{AbstractData}
    time::Bool
end

function Mutator(pop_ids::Vector{String}=String[]; time::Bool=false)
    condition = always
    retriever = PopulationRetriever(pop_ids) # returns vec{vec{pop}}
    operator = noop
    updater = map(map((s,p)->mutate!(s, p)))
    Mutator(condition, retriever, operator, updater, AbstractData[], time)
end

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
