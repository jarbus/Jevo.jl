export State, run!, first_gen, generation
mutable struct State <: AbstractState
    id::String
    rng::AbstractRNG
    creators::Vector{AbstractCreator}
    operators::Vector{AbstractOperator}
    populations::Vector{AbstractPopulation}
    counters::Vector{AbstractCounter}
    matches::Vector{AbstractMatch}
    metrics::Vector{AbstractMetric}
    data::Vector{AbstractData}    # for extensions
    info::Dict{Any, Any}          # for extensions
end


# Allows specifying state by id, rng, creators, and operators
State(id::String, rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator}) = 
    State(id, rng, creators, operators, AbstractPopulation[], default_counters(), AbstractMatch[], AbstractMetric[], AbstractData[], Dict())

# shorthand to create empty states 
State() = State("", StableRNG(1234), AbstractCreator[], AbstractOperator[])
function State(creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator})
    if get(ENV, "JULIA_TEST_MODE", "false") != "true"
        println("Creating state with seed 1")
    end
    State("", StableRNG(1), creators, operators)
end

function operate!(state::AbstractState) 
    for i in 1:length(state.operators)
        try
            operate!(state, state.operators[i])
        catch e
            println("Error in operator ", i, " ", state.operators[i])
            serialize(state, "error-state.jld")
            # write operator id to file
            open("error-operator.txt", "w") do io
                println(io, state.operators[i])
            end


            rethrow(e)
        end
            

    end
end

generation(state::AbstractState) = find(:type, AbstractGeneration, state.counters) |> value
first_gen(state::AbstractState) = generation(state) == 1

run!(state::State, max_generations::Int) = 
    foreach((_)->operate!(state), generation(state):max_generations)
