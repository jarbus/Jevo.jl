export Mutator, mutate
struct Mutator <: AbstractMutator 
    condition::Function 
    retriever::AbstractRetriever # returns vec{vec{ind}} to mutate
    operator::Function  # returns iterable of mutated individuals,
                        # does not update the state
    updater::AbstractUpdater   # adds mutated individuals to the respective 
                               # populations
end

function Mutator(pop_ids::Vector{String}=String[])
    condition = always
    retriever = PopulationRetriever(pop_ids) # returns vec{vec{pop}}
    operator = map((s,x)->mutate(s, x))
    updater = PopulationUpdater(pop_ids)
    Mutator(condition, retriever, operator, updater)
end

mutate(state::AbstractState, pop::AbstractPopulation) =
    [mutate(state, ind) for ind in pop.individuals]

function mutate(state::AbstractState, ind::AbstractIndividual) 
    new_id, gen = new_id_and_gen(state)
    new_geno = mutate(state, ind.genotype)
    Individual(new_id, gen, [ind.id], new_geno, ind.developer)
end
mutate(::AbstractState, genotype::AbstractGenotype) =
    error("mutate function not implemented for $(typeof(genotype))")
