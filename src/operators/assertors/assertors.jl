export Assertor, PopSizeAssertor

@define_op "Assertor" "AbstractAssertor"

Assertor(;kwargs...) = create_op("Assertor", kwargs...)

@define_op "PopSizeAssertor" "AbstractAssertor"

PopSizeAssertor(size::Int, pop_ids::Vector{String}=String[];kwargs...) = 
    create_op("PopSizeAssertor",
               retriever=PopulationRetriever(pop_ids),
               operator=map(map((s, p)->@assert length(p.individuals) == size));kwargs...)
