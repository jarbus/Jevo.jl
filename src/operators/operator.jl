# ==========OPERATORS===========
# Evolutionary operators (e.g., mutator, performer, evaluator, archiver)
# are independent actions that use/update the state
Base.@kwdef struct Operator <: AbstractOperator
    condition::Function          # returns true if the operator should be applied
    retriever::AbstractRetriever # retrieves iterable of objects to operate on
    operator::Function           # returns iterable of operated objects
    updater::AbstractUpdater     # updates the state
    rng::AbstractRNG
    data::Vector{AbstractData} = AbstractData[] # for extensions and recording metrics
end

function operate!(state::AbstractState, operator::AbstractOperator)
    !operator.condition(state) && return
    objects = operator.retriever(state)
    objects = operator.operator(objects)
    operator.updater(state, objects)
end

noop(x...) = x
