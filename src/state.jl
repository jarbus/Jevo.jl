Base.@kwdef mutable struct State <: AbstractState
    id::String
    generation::Int
    environments::Vector{AbstractEnvironment}
    populations::Vector{AbstractPopulation}
    operators::Vector{AbstractOperator}
    counters::Vector{AbstractCounters} # TODO fill this in
    matches::Vector{AbstractMatch} = AbstractMatch[]
    metrics::Vector{AbstractMetric} = AbstractMetric[]
    data::Vector{AbstractData} = AbstractData[] # for extensions
    info::Dict{Any, Any} = Dict()               # for extensions
end

operate!(state::AbstractState) = foreach(op -> operate!(state, op), state.operators)
