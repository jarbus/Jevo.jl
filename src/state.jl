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

@define_op "GenerationIncrementer"
GenerationIncrementer(;kwargs...) = create_op("GenerationIncrementer",
    updater=(state, _)->(get_counter(AbstractGeneration, state) |> inc!); kwargs...)

# Allows specifying state by id, rng, creators, and operators
function State(id::String, rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator}; counters::Vector{<:AbstractCounter}=default_counters(), populations::Vector{<:AbstractPopulation}=AbstractPopulation[], matches::Vector{<:AbstractMatch}=AbstractMatch[], metrics::Vector{<:AbstractMetric}=AbstractMetric[], data::Vector{<:AbstractData}=AbstractData[], info::Dict{Any, Any}=Dict())
    operators = AbstractOperator[operators..., GenerationIncrementer()]
    State(id, rng, creators, operators, populations, counters, matches, metrics, data, info)
end

# shorthand to create empty states 
State() = State("", StableRNG(1234), AbstractCreator[], AbstractOperator[])
function State(rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator})
    if get(ENV, "JULIA_TEST_MODE", "false") != "true"
        println("Creating state with seed 1")
    end
    State("", rng, creators, operators)
end

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


run!(state::State, max_generations::Int) = 
    foreach((_)->operate!(state), generation(state):max_generations)
