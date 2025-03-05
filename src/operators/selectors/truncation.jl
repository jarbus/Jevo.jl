export TruncationSelector, ClusterTruncationSelector
"""
    TruncationSelector <: AbstractSelector

Selects the top k individuals from a population based on their fitness scores, purging the rest. The individuals are sorted by their fitness scores in descending order, and the top k individuals are selected. If there are fewer than k individuals in the population, raises an error.
"""
@define_op "TruncationSelector" "AbstractSelector"
TruncationSelector(k::Int, ids::Vector{String}=String[]; kwargs...) =
    create_op("TruncationSelector",
                    retriever=PopulationRetriever(ids),
                    updater=map((s,p)->truncate!(s,p,k))
                    ;kwargs...)
function truncate!(state::AbstractState, pops::Vector{Population}, k::Int)
    @assert k > 0                           "k must be greater than 0"
    @assert length(pops) == 1               "Truncation selection can only be applied to a single Population"
    @assert length(pops[1].individuals) > k "Population must have more individuals than k= $k to truncate, $(pops[1].id) has $(length(pops[1].individuals)) individuals."
    scores = Vector{Float64}(undef, length(pops[1].individuals))
    for (i, ind) in enumerate(pops[1].individuals)
        @assert length(ind.records) == 1 "Individuals must have exactly one record"
        scores[i] = ind.records[1].fitness
    end
    pops[1].individuals = pops[1].individuals[sortperm(scores, rev=true)[1:k]]

end

@define_op "ClusterTruncationSelector" "AbstractSelector"
ClusterTruncationSelector(k::Int, ids::Vector{String}=String[]; radius=nothing, kwargs...) =
    create_op("ClusterTruncationSelector",
                    retriever=PopulationRetriever(ids),
                    updater=map(map((s,p)->cluster_truncate!(s,p,k,radius)))
                    ;kwargs...)

function cluster_truncate!(state::AbstractState, pop::Population, k::Int, radius)
    # Find the outcome matrix in the population data
    outcome_idx = findfirst(x -> x isa OutcomeMatrix, pop.data)
    @assert !isnothing(outcome_idx) "OutcomeMatrix not found in population $(pop.id)"
    outcome_matrix = pop.data[outcome_idx].matrix
    # confirm outcome matrix is only 1.0 or 0.0
    @assert all(x -> x == 0.0 || x == 1.0, outcome_matrix) "Outcome matrix must be binary"

    # Each row corresponds to an individual in pop.individuals
    # Each column corresponds to a cluster
    # For each cluster, select the k individuals with the highest record.fitness
    selected_individuals = Vector{Individual}()
    for cluster in 1:size(outcome_matrix, 2)
        cluster_members = findall(x -> x == 1.0, outcome_matrix[:, cluster])
        cluster_individuals = [pop.individuals[i] for i in cluster_members]
        @assert length(cluster_individuals) > 0 "Cluster must have at least one member"
        @assert all(x -> length(x.records) == 1, cluster_individuals) "Individuals must have exactly one record"
        cluster_individuals = sort(cluster_individuals, by = x -> x.records[1].fitness, rev = true)
        append!(selected_individuals, cluster_individuals[1:min(k, length(cluster_individuals))])
    end


    pop.individuals = selected_individuals
end
