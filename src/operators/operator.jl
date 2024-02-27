export ClearInteractionsAndRecords

function operate!(state::AbstractState, operator::AbstractOperator)
    !operator.condition(state) && return
    operator.time && (start = time())
    objects = operator.retriever(state)
    objects = operator.operator(state, objects)
    operator.updater(state, objects)
    operator.time && (@info "Operator $(operator.condition) took $(time()-start) seconds")
end

@define_op "ClearInteractionsAndRecords"
ClearInteractionsAndRecords() = create_op("ClearInteractionsAndRecords",
          retriever=get_individuals,
          updater=map((_,ind)->(empty!(ind.interactions);
                                empty!(ind.records))))
