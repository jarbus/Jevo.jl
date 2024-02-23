export Operator, ClearInteractionsAndRecords
# ==========OPERATORS===========
# Evolutionary operators (e.g., mutator, performer, evaluator, archiver)
# are independent actions that use/update the state
Base.@kwdef struct Operator <: AbstractOperator
    condition::Function          # returns true if the operator should be applied
    retriever::AbstractRetriever # retrieves iterable of objects to operate on
    operator::Function           # returns iterable of operated objects
    updater::AbstractUpdater     # updates the state
    data::Vector{AbstractData} = AbstractData[] # for extensions and recording metrics
end

function operate!(state::AbstractState, operator::AbstractOperator)
    !operator.condition(state) && return
    objects = operator.retriever(state)
    objects = operator.operator(state, objects)
    operator.updater(state, objects)
end

Base.@kwdef struct ClearInteractionsAndRecords <: AbstractOperator
    condition::Function = always
    retriever::Function = get_individuals
    operator::Function = noop
    updater::Function = map((_,ind)->(empty!(ind.interactions);
                                      empty!(ind.records)))
end
