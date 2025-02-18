struct OutcomeMatrix
    matrix::Matrix{Float64}
end
export ComputeOutcomeMatrix
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
    @assert length(populations) == 1
    pop = populations[1]
    # objectives can be arbitrarily large numbers, to create interaction matrix
    # we need to map objectives and players to 1:n

    # First, we create a mapping from id (which can be arbitrarily large) to idx
    test_ids = [test_id for ind in pop.individuals 
                           for int in ind.interactions
                           for test_id in int.other_ids]|> unique
    solution_ids = [ind.id for ind in pop.individuals]
    test_idxs = Dict(test_id => idx for (idx, test_id) in enumerate(test_ids))
    solution_idxs = Dict(sol_id => idx for (idx, sol_id) in enumerate(solution_ids))

    # initialize outcomes matrix to avoid repeated dict lookups when lexicasing
    outcomes = zeros(Float64, length(solution_ids), length(test_ids))
    for ind in pop.individuals, int in ind.interactions, 
        test_id in int.other_ids
        sol_idx = solution_idxs[int.individual_id]
        test_idx = test_idxs[test_id]
        outcomes[sol_idx, test_idx] += int.score
    end
    filter!(x->!isa(x, OutcomeMatrix), pop.data)
    push!(pop.data, OutcomeMatrix(outcomes))
end
