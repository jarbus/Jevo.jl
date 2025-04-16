export State, run!, first_gen, generation
"""
    mutable struct State <: AbstractState
        id::String
        rng::AbstractRNG
        creators::Vector{AbstractCreator}
        operators::Vector{AbstractOperator}
        populations::Vector{AbstractPopulation}
        counters::Vector{AbstractCounter}
        matches::Vector{AbstractMatch}
        data::Vector    # for extensions
    end

    A mutable struct which holds all runtime data for an evolutionary simulation. See [`State(rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator})`](@ref) . each generation.

# Example

```julia
using Jevo, StableRNGs
rng = StableRNG(1)

k = 1
n_dims = 2
n_inds = 2
n_species = 2
n_gens = 10

counters = default_counters()
ng_genotype_creator = Creator(VectorGenotype, (n=n_dims,rng=rng))
ng_developer = Creator(VectorPhenotype)

comp_pop_creator = Creator(CompositePopulation, ("species", [("p\$i", n_inds, ng_genotype_creator, ng_developer) for i in 1:n_species], counters))
env_creator = Creator(CompareOnOne)

state = State("ng_phylogeny", rng,[comp_pop_creator, env_creator],
    [InitializeAllPopulations(),
    AllVsAllMatchMaker(),
    Performer(),
    ScalarFitnessEvaluator(),
    TruncationSelector(k),
    CloneUniformReproducer(n_inds),
    Mutator(),
    ClearInteractionsAndRecords(),
    Reporter(GenotypeSum, console=true)], counters=counters)

run!(state, n_gens)
```
"""
mutable struct State <: AbstractState
    id::String
    rng::AbstractRNG
    creators::Vector{AbstractCreator}
    operators::Vector{AbstractOperator}
    populations::Vector{AbstractPopulation}
    counters::Vector{AbstractCounter}
    matches::Vector{AbstractMatch}
    data::Vector    # for extensions
end

# State(rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator}) =
#     State("", rng, creators, operators)
"""
    State(
        id::String,
        rng::AbstractRNG,
        creators::Vector{<:AbstractCreator},
        operators::Vector{<:AbstractOperator};
        counters::Vector{<:AbstractCounter}=default_counters(),
        populations::Vector{<:AbstractPopulation}=AbstractPopulation[],
        matches::Vector{<:AbstractMatch}=AbstractMatch[],
        data::Vector=[],
    )

States are created from a random number generator, a list of creators, and a list of operators, and usually a list of counters.

`creators` should have at least on population creator and one environment creator.
`operators` should contain an operator for each step of the evolutionary process.
`counters` should contain a generation counter, individual id counter, gene counter, and match counter. All creators/operators should refer to the counter objects in state.

Use [`generation(state)`](@ref) to get the current generation number, initialized to one. The [`GenerationIncrementer`](@ref) operator is automatically appended to the operator list to advance the state to the next generation. Individuals created without any parents are of generation 0.
"""
function State(id::String, rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator}; counters::Vector{<:AbstractCounter}, populations::Vector{<:AbstractPopulation}=AbstractPopulation[], matches::Vector{<:AbstractMatch}=AbstractMatch[], data::Vector=[], checkpoint_interval::Int=-1)
    operators = AbstractOperator[operators..., GenerationIncrementer()]
    if checkpoint_interval > 0
        pushfirst!(operators, LoadCheckpoint("./checkpoint.jls"))
        push!(operators,  Checkpoint("./checkpoint.jls", interval=checkpoint_interval))
    end
    State(id, rng, creators, operators, populations, counters, matches, data)
end

# shorthand to create empty states 
function State(operators::Vector{O}=AbstractOperator[]) where {O <: AbstractOperator}
    State("", StableRNG(1234), AbstractCreator[], operators, counters=default_counters())
end


function operate!(state::AbstractState) 
    for i in 1:length(state.operators)
        try
            if i > 1 && state.operators[i] isa LoadCheckpoint
                @error "Encountered LoadCheckpoint after the first operator; terminating"
            elseif i == 1 && state.operators[1] isa LoadCheckpoint
                operate!(state, state.operators[i]) # load checkpoint
                return
            end
            operate!(state, state.operators[i])
        catch e
            println("Error in operator ", i, " ", state.operators[i], " at generation ", generation(state))
            if get(ENV, "NO_SERIALIZE_ON_ERROR", "0") ∈ ["0", "false"]
                println("Serializing state...")
                serialize("error-state.jld", state)
                # write operator id to file
                open("error-operator.txt", "w") do io
                    println(io, state.operators[i])
                end
            end
            rethrow(e)
        end
        i += 1
    end
end


"""
    run!(state::State, max_generations::Int)

Begin/continue evolution until generation `max_generations`
"""
function run!(state::State, max_generations::Int) 
    for i in generation(state):max_generations
        operate!(state)
    end
    for i in generation(state):max_generations
        operate!(state)
    end
end

"""
    generation(state::AbstractState)

Return the current generation number of the state stored in the `AbstractGeneration` counter.
"""
generation(state::AbstractState) = find(:type, AbstractGeneration, state.counters) |> value

"""
    first_gen(state::AbstractState)

Return true if the current generation is 1.
"""
first_gen(state::AbstractState) = generation(state) == 1

"""
Condition for operators that should run each generation
"""
always(state::AbstractState) = true
