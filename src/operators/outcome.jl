using Clustering
using Plots

struct OutcomeMatrix
    matrix::Matrix{Float64}
end
export ComputeOutcomeMatrix, ClusterOutcomeMatrix
"""
    ComputeOutcomeMatrix(ids::Vector{String}=String[]; kwargs...)

"""
@define_op "ComputeOutcomeMatrix" "AbstractEvaluator"
ComputeOutcomeMatrix(ids::Vector{String}=String[]; kwargs...) =
    create_op("ComputeOutcomeMatrix",
            retriever=PopulationRetriever(ids),
            updater=map(add_outcome_matrices!); kwargs...)

function add_outcome_matrices!(::AbstractState,
        populations::Vector{<:AbstractPopulation})
    # confirm there are no outcomes already
    for pop in populations
        @assert !any(x->x isa OutcomeMatrix, pop.data) "OutcomeMatrix already exists in population $(pop.id)"
    end
    @assert length(populations) == 1
    pop = populations[1]
    # objectives can be arbitrarily large numbers, to create interaction matrix
    # we need to map objectives and players to 1:n

    # First, we create a mapping from id (which can be arbitrarily large) to idx
    test_ids = [test_id for ind in pop.individuals 
                            for int in ind.interactions if (int isa Interaction || int isa EstimatedInteraction)
                           for test_id in int.other_ids]|> unique

    solution_ids = [ind.id for ind in pop.individuals]
    @assert length(test_ids) > 0
    @assert length(solution_ids) > 0
    test_idxs = Dict(test_id => idx for (idx, test_id) in enumerate(test_ids))
    solution_idxs = Dict(sol_id => idx for (idx, sol_id) in enumerate(solution_ids))

    # initialize outcomes matrix to avoid repeated dict lookups when lexicasing
    outcomes = zeros(Float64, length(solution_ids), length(test_ids))

    outcome_entered = fill(false, length(solution_ids), length(test_ids))
    for ind in pop.individuals, int in ind.interactions, test_id in int.other_ids
        !(int isa Interaction || int isa EstimatedInteraction) && continue
        sol_idx = solution_idxs[int.individual_id]
        test_idx = test_idxs[test_id]
        outcomes[sol_idx, test_idx] += int.score
        outcome_entered[sol_idx, test_idx] = true
    end
    @assert all(outcome_entered) "Not all outcomes were entered into the matrix $outcome_entered"
    #filter!(x->!isa(x, OutcomeMatrix), pop.data)
    push!(pop.data, OutcomeMatrix(outcomes))
end



@define_op "ClearOutcomeMatrix" "AbstractOperator"
ClearOutcomeMatrix(ids::Vector{String}=String[]; kwargs...) =
    create_op("ClearOutcomeMatrix",
                    retriever=PopulationRetriever(ids),
                    updater=map(map((s,p)->filter!(x->!isa(x, OutcomeMatrix), p.data))),
                    ;kwargs...)

function clear_missing_interactions!(state, pops)
    @assert length(pops) == 1 && length(pops[1]) == 1 "Only one population with one subpopulation is supported for ClearMissingInteractions"
    pop = pops[1][1]
    # confirm all invidivudals have at least one interaction
    #@assert all(length(ind.interactions) > 0 for ind in pop.individuals)
    ind_ids = Set(ind.id for ind in pop.individuals)
    # remove any interaction where all ids in interation.other_ids is not in ind_ids
    for ind in pop.individuals 
        filter!(int->all(id->id in ind_ids, int.other_ids), ind.interactions)
    end
end

@define_op "ClearMissingInteractions" "AbstractOperator"
ClearMissingInteractions(ids::Vector{String}=String[]; kwargs...) =
    create_op("ClearMissingInteractions",
                    retriever=PopulationRetriever(ids),
                    updater=(s,p)->clear_missing_interactions!(s,p),
                    ;kwargs...)

"""
    ClusterOutcomeMatrix(ids::Vector{String}=String[]; eps=0.5, min_points=5, kwargs...)

Clusters the outcome matrix using DBScan algorithm and creates a new outcome matrix
where each row is assigned to a cluster. The new matrix has dimensions (rows × k),
where k is the number of discovered clusters. Each element is 1 if the row belongs
to that cluster, and 0 otherwise.

Parameters:
- `eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other
- `min_points`: The number of samples in a neighborhood for a point to be considered as a core point
"""
@define_op "ClusterOutcomeMatrix" "AbstractOperator"
ClusterOutcomeMatrix(ids::Vector{String}=String[]; k::Int, kwargs...) =
    create_op("ClusterOutcomeMatrix",
            retriever=PopulationRetriever(ids),
            updater=map(map((s, p)  -> cluster_outcome_matrices!(s, p, k ))),
            ; kwargs...)

function cluster_outcome_matrices!(state::AbstractState, pop::Population, k)
    
    # Find the outcome matrix in the population data
    outcome_idx = findfirst(x -> x isa OutcomeMatrix, pop.data)
    @assert !isnothing(outcome_idx) "OutcomeMatrix not found in population $(pop.id)"
    outcome_matrix = pop.data[outcome_idx].matrix
    deleteat!(pop.data, outcome_idx)
    # outcomes are solutions x tests
    # transpose to get samples as columns
    #samples = outcome_matrix |> transpose 
    samples = outcome_matrix
    # samples are tests x solutions

    # min-max normalization along each row, handle division by zero
    #samples .-= minimum(samples, dims=2)
    #samples ./= maximum(samples, dims=2) .+ 1e-6

    # save as heatmap before clustering
    if generation(state) % 100 == 0
        heatmap(samples, aspect_ratio=1, title="Outcome Matrix Before Clustering", xlabel="Individual", ylabel="Test")
        savefig("media/outcome_matrix_before_clustering.png")
    end

    n_samples = size(samples, 1)
    n_tests = size(samples, 2)
    
    n_k_means = 256
    results = Vector{Clustering.KmeansResult}(undef, n_k_means)
    Threads.@threads for i in 1:n_k_means
        results[i] = kmeans(samples, k, init=:rand)
    end
    # Find the best clustering result
   result = argmin(x -> x.totalcost, results)
    
    # Get the number of clusters (excluding noise points marked as 0)
    clusters = unique(result.assignments)
    if 0 in clusters  # Remove noise cluster (marked as 0)
        clusters = filter(c -> c != 0, clusters)
    end
    # print number of points in each cluster
    max_cluster_size = 0
    for j in 1:k
        cluster_size = count(x -> x == j, result.assignments)
        max_cluster_size = max(max_cluster_size, cluster_size)
        if cluster_size > 1
            @info "Cluster $j: $cluster_size points"
        end
    end

    m =  Measurement("max_cluster_size", max_cluster_size, generation(state))
    @info m
    @h5 m
    
    # Create a new outcome matrix based on cluster assignments
    # Each point corresponds to the sum of all tests within each cluster
    n_rows = size(outcome_matrix, 1)
    cluster_matrix = zeros(Float64, n_rows, k)
    
    for i in 1:n_rows
        for j in 1:n_tests
            cluster_idx = result.assignments[j]
            if cluster_idx != 0  # Exclude noise points
                cluster_matrix[i, cluster_idx] += outcome_matrix[i, j]
            end
        end
    end
    
    # Normalize the cluster matrix by the number of points in each cluster
    for j in 1:k
        cluster_size = count(x -> x == j, result.assignments)
        if cluster_size > 0
            cluster_matrix[:, j] ./= cluster_size
        end
    end

    # save as heatmap after clustering
    if generation(state) % 100 == 0
        heatmap(cluster_matrix, aspect_ratio=1, title="Clustered Outcome Matrix", xlabel="Cluster", ylabel="Individual")
        savefig("media/clustered_outcome_matrix.png")
    end
    
    # Add the clustered outcome matrix to population data
    push!(pop.data, OutcomeMatrix(cluster_matrix))
end
