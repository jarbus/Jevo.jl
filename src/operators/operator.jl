export ClearInteractionsAndRecords, Operator, @define_op, create_op

"""
    Op
"""
struct Operator <: AbstractOperator end

function operate!(state::AbstractState, operator::AbstractOperator)
    !operator.condition(state) && return
    operator.time && (start = time())
    objects = operator.retriever(state, operator)
    objects = operator.operator(state, objects)
    operator.updater(state, objects)
    operator.time && (@info "Operator $(typeof(operator)) took $(round((time()-start),digits=4)) seconds")
end


"""
    @define_op name [supertype=AbstractOperator, additional_fields=""]

Defines a a subtype of `supertype` called `name` with the following fields:

- `condition`
- `retriever`
- `operator`
- `updater`
- `data`
- `time`
"""
macro define_op(name, supertype="AbstractOperator", additional_fields="")
    field_defs = [
        :(condition::Function),
        :(retriever::Union{Function, AbstractRetriever}),
        :(operator::Function),
        :(updater::Union{Function, AbstractUpdater}),
        :(data::Vector{<:AbstractData}),
        :(time::Bool),
    ]
    if additional_fields != ""
        additional_defs = split(additional_fields, ",")
        for field_def in additional_defs
            name_type = split(field_def, ":")
            push!(field_defs, :($(Symbol(name_type[1]))::$(Symbol(name_type[2]))))
        end
    end
    esc(
        :(struct $(Symbol(name)) <: $(Symbol(supertype))
            $(field_defs...)
        end)
    )
end

"""
"""
function create_op(type::Type{<:AbstractOperator};
        condition=always,
        retriever=noop,
        operator=noop,
        updater=noop,
        data=AbstractData[],
        time=false,
        additional_fields...)
    # Convert symbol to type
    @assert type <: AbstractOperator
    type(condition, retriever, operator, updater, data, time; additional_fields...)
end

create_op(name::String; kwargs...) = create_op(eval(Symbol(name)); kwargs...)

@define_op "ClearInteractionsAndRecords"
ClearInteractionsAndRecords(;kwargs...) = create_op("ClearInteractionsAndRecords",
          retriever=get_individuals,
          updater=map((_,ind)->reset_individual!(ind)); kwargs...)

@define_op "GenerationIncrementer"

"""
    GenerationIncrementer(;kwargs...)

Increments the generation counter in the state by 1.
"""
GenerationIncrementer(;kwargs...) = create_op("GenerationIncrementer",
    updater=(state, _)->(get_counter(AbstractGeneration, state) |> inc!); kwargs...)
