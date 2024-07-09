export Assertor, PopSizeAssertor

@define_op "Assertor" "AbstractAssertor"

"""    
    Assertor(;kwargs...)

Creates an [Operator](@ref) that asserts some condition. You probably want to override the `retriever` `operator` fields.
"""
Assertor(;kwargs...) = create_op("Assertor", kwargs...)

"""
    PopSizeAssertor(size::Int, pop_ids::Vector{String}=String[];kwargs...)

Asserts each population with an `id` in `pop_ids` has `size` individuals.
"""
@define_op "PopSizeAssertor" "AbstractAssertor"

PopSizeAssertor(size::Int, pop_ids::Vector{String}=String[];kwargs...) = 
    create_op("PopSizeAssertor",
               retriever=PopulationRetriever(pop_ids),
               operator=map(map((s, p)->@assert length(p.individuals) == size));kwargs...)
