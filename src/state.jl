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
        data::Vector{AbstractData}    # for extensions
    end


A mutable struct which holds all runtime data for an evolutionary simulation.


"""
mutable struct State <: AbstractState
    id::String
    rng::AbstractRNG
    creators::Vector{AbstractCreator}
    operators::Vector{AbstractOperator}
    populations::Vector{AbstractPopulation}
    counters::Vector{AbstractCounter}
    matches::Vector{AbstractMatch}
    data::Vector{AbstractData}    # for extensions
end

"""
    State(rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator})

States are created from a random number generator, a list of creators, and a list of operators. The [`GenerationIncrementer`](@ref) operator is automatically appended to the operator list to advance the state to the next generation.
"""
State(rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator}) =
    State("", rng, creators, operators)

function State(id::String, rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator}; counters::Vector{<:AbstractCounter}=default_counters(), populations::Vector{<:AbstractPopulation}=AbstractPopulation[], matches::Vector{<:AbstractMatch}=AbstractMatch[], data::Vector{<:AbstractData}=AbstractData[], )
    operators = AbstractOperator[operators..., GenerationIncrementer()]
    State(id, rng, creators, operators, populations, counters, matches, data)
end

# shorthand to create empty states 
State() = State("", StableRNG(1234), AbstractCreator[], AbstractOperator[])


function operate!(state::AbstractState) 
    for i in 1:length(state.operators)
        try
            operate!(state, state.operators[i])
        catch e
            println("Error in operator ", i, " ", state.operators[i])
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
    end
end


"""
    run!(state::State, max_generations::Int)

Begin/continue evolution until generation `max_generations`
"""
run!(state::State, max_generations::Int) = 
    foreach((_)->operate!(state), generation(state):max_generations)

generation(state::AbstractState) = find(:type, AbstractGeneration, state.counters) |> value
first_gen(state::AbstractState) = generation(state) == 1
always(state::AbstractState) = true
