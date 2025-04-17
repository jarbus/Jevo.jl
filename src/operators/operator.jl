export ClearInteractionsAndRecords, Operator, @define_op, create_op


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

See [`Operator`](@ref) for more information.

# Example

```julia
@define_op "AllVsAllMatchMaker" "AbstractMatchMaker"
```
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
    Operator <: AbstractOperator

All operator structs should be defined with the [`@define_op`](@ref) macro, and all operators should be created with the [`create_op`](@ref) function. Each operator should have the following fields:

    condition::Function
    retriever::Union{Function, AbstractRetriever}
    operator::Function
    updater::Union{Function, AbstractUpdater}
    data::Vector{<:AbstractData}
    time::Bool


`condition` is a function that takes the state and returns a boolean. If `condition(state) == false`, the operator will not be executed. Ex: [`always`](@ref), [`first_gen`](@ref)

`retriever` is a function/object that takes the state and the operator and returns an object or list of objects to operate on. Should not update state. Ex: [`get_individuals`](@ref), [`PopulationRetriever`](@ref)

`operator` is a function that takes the state and the retrieved object(s) and returns the modified object(s). Should perform none or trivial state updates, like incrementing a counter. Ex: [`Jevo.make_all_v_all_matches`](@ref)

`updater` is a function/object that takes the state and the modified object(s) and updates the state. Ex: [`ReccordAdder!`](@ref), [`ComputeInteractions!`](@ref)

`data` is a vector of data objects that can be stored in the operator. Currently unused.

`time` is a boolean that determines if the time taken to execute the operator should be logged. Defaults to `false`
"""
@define_op "Operator"

"""
    create_op(
        type::Union{Type{<:AbstractOperator}, String};
        condition=always,
        retriever=noop,
        operator=noop,
        updater=noop,
        data=AbstractData[],
        time=false,
        additional_fields...
    )

Create an [Operator](@ref) of type `type` with the specified fields. All fields are optional. The retriever, operator, and updater fields default to [`noop`](@ref).
"""
function create_op(type::Type{<:AbstractOperator}, additional_fields...;
        condition=always,
        retriever=noop,
        operator=noop,
        updater=noop,
        data=AbstractData[],
                   time=false,)
        #additional_fields...)
    # Convert symbol to type
    @assert type <: AbstractOperator
    type(condition, retriever, operator, updater, data, time, additional_fields...)
end

create_op(name::String, additional_fields...; kwargs...) = create_op(eval(Symbol(name)), additional_fields...; kwargs...)

"""
    ClearInteractionsAndRecords(;kwargs...)

Clears all interactions and records from all individuals in the state.
"""
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
