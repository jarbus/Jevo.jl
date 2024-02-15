Base.@kwdef mutable struct State <: AbstractState
    id::String
    creators::Vector{AbstractCreator}
    operators::Vector{AbstractOperator}
    environments::Vector{AbstractEnvironment}
    populations::Vector{AbstractPopulation}
    counters::Vector{AbstractCounter} = default_counters()
    matches::Vector{AbstractMatch} = AbstractMatch[]
    metrics::Vector{AbstractMetric} = AbstractMetric[]
    data::Vector{AbstractData} = AbstractData[] # for extensions
    info::Dict{Any, Any} = Dict()               # for extensions
    rng::AbstractRNG
end

operate!(state::AbstractState) = foreach(op -> operate!(state, op), state.operators)
