export @define_op, create_op

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
