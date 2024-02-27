export ScalarFitnessEvaluator
@define_op "ScalarFitnessEvaluator" "AbstractEvaluator"
ScalarFitnessEvaluator(ids::Vector{String}=String[]; kwargs...) =
    create_op("ScalarFitnessEvaluator",
            retriever=PopulationRetriever(ids),
            operator=map(make_scalar_fitness_records),
            updater=RecordAdder(ids); kwargs...)

function make_scalar_fitness_records(::AbstractState,
        population::Vector{<:AbstractPopulation})
    num_inds = 0
    min_score = Inf
    scores = Vector{Float32}[]
    records = Vector{Vector{<:AbstractRecord}}()
    # Compute scores
    for subpop in population
        push!(scores, Float32[])
        for ind in subpop.individuals
            num_inds += 1
            score = mean(interaction.score for interaction in ind.interactions)
            push!(scores[end], score)
            min_score = min(min_score, score)
        end
    end
    # Compute score shifted by min and create records
    @assert min_score != Inf "No valid interactions found"
    for (subpop, subpop_scores) in zip(population, scores)
        @assert length(subpop.individuals) == length(subpop_scores)
        push!(records, Record[])
        for (ind, score) in zip(subpop.individuals, subpop_scores)
            push!(records[end], Record(ind.id, score-min_score))
        end
        @assert length(subpop.individuals) == length(records[end])
    end
    @assert length(population) == length(records)
    records
end
