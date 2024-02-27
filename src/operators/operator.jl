export ClearInteractionsAndRecords

function operate!(state::AbstractState, operator::AbstractOperator)
    !operator.condition(state) && return
    operator.time && (start = time())
    objects = operator.retriever(state)
    objects = operator.operator(state, objects)
    operator.updater(state, objects)
    operator.time && (@info "Operator $(typeof(operator)) took $(round((time()-start),digits=4)) seconds")
end

@define_op "ClearInteractionsAndRecords"
ClearInteractionsAndRecords(;kwargs...) = create_op("ClearInteractionsAndRecords",
          retriever=get_individuals,
          updater=map((_,ind)->reset_individual!(ind)))
