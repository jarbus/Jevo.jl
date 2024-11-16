export ScalarFitnessEvaluator
"""
    ScalarFitnessEvaluator(ids::Vector{String}=String[]; kwargs...)

[Operator](@ref) that computes the scalar fitness of each individual in populations with ids in `ids`. Requires all individuals to have at least one interaction.

The scalar fitness is the mean of the scores of each interaction, shifted by the minimum score. 
"""
@define_op "ScalarFitnessEvaluator" "AbstractEvaluator"
ScalarFitnessEvaluator(ids::Vector{String}=String[]; kwargs...) =
    create_op("ScalarFitnessEvaluator",
            retriever=PopulationRetriever(ids),
            operator=map(make_scalar_fitness_records),
            updater=ReccordAdder!(ids); kwargs...)

function make_scalar_fitness_records(::AbstractState,
        population::Vector{<:AbstractPopulation})
    @assert length(population) > 0 "Must be at least one population"
    num_inds = 0
    min_score = Inf
    scores = Vector{Float64}[]
    records = Vector{Vector{<:AbstractRecord}}()
    # Compute scores
    for subpop in population
        push!(scores, Float64[])
        @assert length(subpop.individuals) > 0 "Subpopulation must have individuals"
        for ind in subpop.individuals
            num_inds += 1
            @assert length(ind.interactions) > 0 "Individual $(ind.id) must have interactions"
            score = sum(interaction.score for interaction in ind.interactions)
            #= @assert -Inf < score < Inf =#
            push!(scores[end], score)
            min_score = min(min_score, score)
        end
    end
    # Compute score shifted by min and create records
    #= @assert min_score != Inf && min_score != -Inf "No valid interactions found" =#
    for (subpop, subpop_scores) in zip(population, scores)
        @assert length(subpop.individuals) == length(subpop_scores)
        push!(records, Record[])
        for (ind, score) in zip(subpop.individuals, subpop_scores)
            shifted_score = isinf(min_score) ? score : score-min_score
            push!(records[end], Record(ind.id, shifted_score))
        end
        @assert length(subpop.individuals) == length(records[end])
    end
    @assert length(population) == length(records)
    records
end
