using Jevo: Interaction, OutcomeMatrix

@testset "DBScan clustering" begin
    # Create a population with individuals that have interactions
    id_counter = Counter(AbstractIndividual)
    ng_developer = Creator(VectorPhenotype)

    # Create genotypes with distinct patterns to form clear clusters
    genotypes = [
        VectorGenotype([1.0, 0.0]),  # Cluster 1
        VectorGenotype([0.9, 0.1]),  # Cluster 1
        VectorGenotype([0.0, 1.0]),  # Cluster 2
        VectorGenotype([0.1, 0.9]),  # Cluster 2
        VectorGenotype([0.5, 0.5])   # Potential noise point
    ]

    inds = [Individual(inc!(id_counter), 0, Int[], genotypes[i], ng_developer) for i in 1:5]

    # Create population and state
    pop = Population("test_pop", inds)

    outcome_matrix = [1.0 0.9 0.1 0.2 0.5;
                      0.9 1.0 0.2 0.1 0.4;
                      0.1 0.2 1.0 0.9 0.5;
                      0.2 0.1 0.9 1.0 0.6;
                      0.5 0.4 0.5 0.6 1.0]

    push!(pop.data, OutcomeMatrix(outcome_matrix))
    state = State()
    
    # Apply DBScan clustering with parameters that should identify 2 clusters
    # eps=0.3 and min_points=2 should group (1,2) and (3,4) into separate clusters
    # and potentially mark individual 5 as noise
    Jevo.cluster_outcome_matrices!(state, pop, 0.3)
    
    # There should still be one outcome matrices
    outcome_matrices = filter(x -> x isa OutcomeMatrix, pop.data)
    @test length(outcome_matrices) == 1
    
    clustered_matrix = outcome_matrices[1].matrix
    
    @test size(clustered_matrix) == (5, 3)

    # confirm that first column of cluster matrix is the sum of the first two columns
    @test clustered_matrix[:, 1] == (outcome_matrix[:, 1] .+ outcome_matrix[:, 2]) ./ 2
    # confirm that second column of cluster matrix is the sum of the third and fourth columns
    @test clustered_matrix[:, 2] == (outcome_matrix[:, 3] .+ outcome_matrix[:, 4]) ./ 2
    # confirm that third column of cluster matrix is the fifth column
    @test clustered_matrix[:, 3] == outcome_matrix[:, 5]
    
end

