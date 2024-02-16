export State
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
State(id::String, rng::AbstractRNG, creators::Vector{AbstractCreator}, operators::Vector{AbstractOperator}) = 
    State(id, rng, creators, operators, AbstractPopulation[], default_counters(), AbstractMatch[], AbstractMetric[], AbstractData[], Dict())

operate!(state::AbstractState) = foreach(op -> operate!(state, op), state.operators)
