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

generation(state::AbstractState) = find(:type, AbstractGeneration, state.counters) |> value
first_gen(state::AbstractState) = generation(state) == 1
always(state::AbstractState) = true

Base.@kwdef struct GenerationIncrementer <: AbstractOperator
    condition = always
    retriever = noop
    operator = noop 
    updater::Function = (state, kwargs...) -> get_counter(AbstractGeneration, state) |> inc!
    data::Vector{AbstractData} = AbstractData[]
end


# Allows specifying state by id, rng, creators, and operators
# TODO: add automatic generation incrementer at the end of the operators
function State(id::String, rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator})
    operators = AbstractOperator[operators..., GenerationIncrementer()]
    State(id, rng, creators, operators, AbstractPopulation[], default_counters(), AbstractMatch[], AbstractMetric[], AbstractData[], Dict())
end

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
            println("Error in operator ", i, " ", state.operators[i], ", serializing state.")
            serialize("error-state.jld", state)
            # write operator id to file
            open("error-operator.txt", "w") do io
                println(io, state.operators[i])
            end
            rethrow(e)
        end
    end
end


run!(state::State, max_generations::Int) = 
    foreach((_)->operate!(state), generation(state):max_generations)
