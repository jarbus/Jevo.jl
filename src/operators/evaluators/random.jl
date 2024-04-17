export RandomEvaluator
@define_op "RandomEvaluator" "AbstractEvaluator"
RandomEvaluator(ids::Vector{String}=String[]; kwargs...) =
    create_op("RandomEvaluator",
            retriever=PopulationRetriever(ids),
            operator=map(make_random_records),
            updater=RecordAdder(ids); kwargs...)

function make_random_records(::AbstractState,
        populations::Vector{<:AbstractPopulation})
    records = Vector{Vector{<:AbstractRecord}}()
    for pop in populations
        push!(records, Record[])
        for ind in pop.individuals
            push!(records[end], Record(ind.id, rand()))
        end
        @assert length(pop.individuals) == length(records[end])
    end
    @assert length(populations) == length(records)
    records
end
