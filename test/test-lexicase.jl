using Jevo: Interaction, OutcomeMatrix

@testset "lexicase units" begin
    rng = StableRNG(1)
    pop_size = 4
    n_samples = 1000
    @testset "best_ind" begin
        outcomes = fill(-1.0, pop_size, pop_size)
        outcomes[1,:] .= -0.5
        ϵ = zeros(pop_size)
        @test all([Jevo.lexicase_sample(rng, outcomes, ϵ) == 1 for i in 1:n_samples])
        ϵ .= 2.0
        idxs = [Jevo.lexicase_sample(rng, outcomes, ϵ) for _ in 1:n_samples]
        for i in 1:pop_size  # confirm there's at least 12.5% of each idx, expected 25% bc of ϵ
            @test count(x->x==i, idxs) > n_samples / 8
        end
    end
    @testset "two pareto inds and one epsilon ind" begin
        outcomes = zeros(pop_size, pop_size)
        ϵ = fill(0.5, pop_size)
        outcomes[1,1:2] .= 1.0   #    44% chance (33 + 11)
        outcomes[2,2:3] .= 1.0   #    28% chance (11 + 16.7)
        outcomes[3,2:3] .= 0.99  #    28% chance (11 + 16.7)
        idxs = [Jevo.lexicase_sample(rng, outcomes, ϵ) for _ in 1:n_samples]
        @test count(x->x==1, idxs) > n_samples * 0.40
        @test count(x->x==2, idxs) > n_samples * 0.22
        @test count(x->x==3, idxs) > n_samples * 0.22
    end

end

@testset "lexicase_select!" begin
    id_counter = Counter(AbstractIndividual)
    ng_developer = Creator(VectorPhenotype)

    genotypes = [VectorGenotype([1.0, 0.0]),
                 VectorGenotype([0.0, 1.0]),
                 VectorGenotype([0.0, 0.0])]

    blank_inds = [Individual(inc!(id_counter), 0, Int[], genotypes[i], ng_developer) for i in 1:3]

    # 1 is best on 2, 2 is best on 1, 3 is best on 1
    # not how all numbers games work, but this is just a test
    pareto_front_inds = deepcopy(blank_inds)
    # create interactions
    pareto_front_inds[1].interactions = [
        Interaction(1, [1],0),
        Interaction(1, [2],1),
        Interaction(1, [3],1),
    ]
    pareto_front_inds[2].interactions = [
        Interaction(2, [1],1),
        Interaction(2, [2],0),
        Interaction(2, [3],1),
    ]
    pareto_front_inds[3].interactions = [
        Interaction(3, [1],0),
        Interaction(3, [2],0),
        Interaction(3, [3],0),
    ]

    @testset "selection_only" begin
        state = State()
        pop = Population("p", deepcopy(pareto_front_inds))
        # args: state, pops, pop_size, ϵ, elitism, selection_only, keep_all_parents
        Jevo.add_outcome_matrices!(state, [pop])
        Jevo.lexicase_select!(state, pop, 100, false, true, true, false)
        @test length(pop.individuals) == 2  # ind 3 should be filtered
        @test all(ind->ind.id in [1,2], pop.individuals)
    end

    @testset "keep_parents" begin
        state = State()
        pop = Population("p", deepcopy(pareto_front_inds))
        # args: state, pops, pop_size, ϵ, elitism, selection_only, keep_all_parents
        Jevo.add_outcome_matrices!(state, [pop])
        Jevo.lexicase_select!(state, pop, 100, false, true, false, true)
        # confirm first three inds are the same
        for i in 1:3
            @test pop.individuals[i].id === pareto_front_inds[i].id
        end
        # confirm all remaining pop members have either 1 or 2 as a parent
        for i in 4:100
            @test pop.individuals[i].parents[1] in [1,2]
        end
    end
end

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
    
    # Create interactions to form a clear pattern in the outcome matrix
    # Cluster 1 individuals (1,2) perform similarly
    # Cluster 2 individuals (3,4) perform similarly
    # Individual 5 is different from both clusters
    
    # Individual 1 (Cluster 1)
    inds[1].interactions = [
        Interaction(1, [1], 1.0),
        Interaction(1, [2], 0.9),
        Interaction(1, [3], 0.1),
        Interaction(1, [4], 0.2),
        Interaction(1, [5], 0.5)
    ]
    
    # Individual 2 (Cluster 1)
    inds[2].interactions = [
        Interaction(2, [1], 0.9),
        Interaction(2, [2], 1.0),
        Interaction(2, [3], 0.2),
        Interaction(2, [4], 0.1),
        Interaction(2, [5], 0.4)
    ]
    
    # Individual 3 (Cluster 2)
    inds[3].interactions = [
        Interaction(3, [1], 0.1),
        Interaction(3, [2], 0.2),
        Interaction(3, [3], 1.0),
        Interaction(3, [4], 0.9),
        Interaction(3, [5], 0.5)
    ]
    
    # Individual 4 (Cluster 2)
    inds[4].interactions = [
        Interaction(4, [1], 0.2),
        Interaction(4, [2], 0.1),
        Interaction(4, [3], 0.9),
        Interaction(4, [4], 1.0),
        Interaction(4, [5], 0.6)
    ]
    
    # Individual 5 (potential noise point)
    inds[5].interactions = [
        Interaction(5, [1], 0.5),
        Interaction(5, [2], 0.4),
        Interaction(5, [3], 0.5),
        Interaction(5, [4], 0.6),
        Interaction(5, [5], 1.0)
    ]
    
    # Create population and state
    pop = Population("test_pop", inds)
    state = State()
    
    # First compute the outcome matrix
    Jevo.add_outcome_matrices!(state, [pop])
    
    # Verify the outcome matrix was created
    @test any(x -> x isa OutcomeMatrix, pop.data)
    outcome_matrix = getonly(x -> x isa OutcomeMatrix, pop.data)
    @test size(outcome_matrix.matrix) == (5, 5)
    
    # Apply DBScan clustering with parameters that should identify 2 clusters
    # eps=0.3 and min_points=2 should group (1,2) and (3,4) into separate clusters
    # and potentially mark individual 5 as noise
    Jevo.cluster_outcome_matrices!(state, pop, 0.3, 2)
    
    # There should now be two outcome matrices
    outcome_matrices = filter(x -> x isa OutcomeMatrix, pop.data)
    @test length(outcome_matrices) == 2
    
    # Get the clustered matrix
    clustered_matrix = outcome_matrices[2].matrix
    
    # The clustered matrix should have 5 rows (one per individual)
    # and 2 columns (one per cluster)
    @test size(clustered_matrix, 1) == 5
    
    # Check that individuals 1 and 2 are in the same cluster
    # and individuals 3 and 4 are in a different cluster
    cluster1_idx = findfirst(j -> clustered_matrix[1, j] == 1.0, 1:size(clustered_matrix, 2))
    cluster2_idx = findfirst(j -> clustered_matrix[3, j] == 1.0, 1:size(clustered_matrix, 2))
    
    @test !isnothing(cluster1_idx)
    @test !isnothing(cluster2_idx)
    @test cluster1_idx != cluster2_idx
    
    # Check that individual 2 is in the same cluster as individual 1
    @test clustered_matrix[2, cluster1_idx] == 1.0
    
    # Check that individual 4 is in the same cluster as individual 3
    @test clustered_matrix[4, cluster2_idx] == 1.0
    
    # Test the operator version
    pop = Population("test_pop", deepcopy(inds))
    state = State()
    
    # First compute the outcome matrix
    Jevo.add_outcome_matrices!(state, [pop])
    
    # Create and apply the ClusterOutcomeMatrix operator
    cluster_op = ClusterOutcomeMatrix(eps=0.3, min_points=2)
    apply!(state, cluster_op)
    
    # Verify results
    outcome_matrices = filter(x -> x isa OutcomeMatrix, pop.data)
    @test length(outcome_matrices) == 2
    
    # Get the clustered matrix
    clustered_matrix = outcome_matrices[2].matrix
    
    # The clustered matrix should have 5 rows (one per individual)
    # and 2 columns (one per cluster)
    @test size(clustered_matrix, 1) == 5
    
    # Check that individuals 1 and 2 are in the same cluster
    # and individuals 3 and 4 are in a different cluster
    cluster1_idx = findfirst(j -> clustered_matrix[1, j] == 1.0, 1:size(clustered_matrix, 2))
    cluster2_idx = findfirst(j -> clustered_matrix[3, j] == 1.0, 1:size(clustered_matrix, 2))
    
    @test !isnothing(cluster1_idx)
    @test !isnothing(cluster2_idx)
    @test cluster1_idx != cluster2_idx
    
    # Check that individual 2 is in the same cluster as individual 1
    @test clustered_matrix[2, cluster1_idx] == 1.0
    
    # Check that individual 4 is in the same cluster as individual 3
    @test clustered_matrix[4, cluster2_idx] == 1.0
end

#= @testset "numbers game lexicase" begin =#
#=   n_dims = 2 =#
#=   n_inds = 100 =#
#=   n_species = 2 =#
#=   counters = default_counters() =#
#=   ng_gc = Creator(VectorGenotype, (n=n_dims,rng=rng)) =#
#=   ng_developer = Creator(VectorPhenotype) =#
#=   comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_gc, ng_developer) for i in 1:n_species], counters)) =#
#=   env_creator = Creator(CompareOnOne) =#
#=   isfile("statistics.h5") && rm("statistics.h5") =#
#=   min_sum_computer = create_op("Reporter", =#
#=                         operator=(s,_) -> measure(GenotypeSum, s, false,false,false).min) =#
#=   with_logger(JevoLogger()) do =#
#=       state = State("", rng,[comp_pop_creator, env_creator], =#
#=                     [InitializeAllPopulations(),  =#
#=                     AllVsAllMatchMaker(["p1", "p2"]), =#
#=                     Performer(time=true), =#
#=                     ComputeOutcomeMatrix(), =#
#=                     ElitistLexicaseSelectorAndReproducer(n_inds, ϵ=true, time=true), =#
#=                     min_sum_computer, =#
#=                     Reporter(GenotypeSum), =#
#=                     Mutator(), =#
#=                     ClearInteractionsAndRecords(), ], counters=counters) =#
#=       run!(state, 5) =#
#=   end =#
#= end =#
#==#
#= @testset "repeat sequence tfr lexicase" begin =#
#=     n_inds = 3 =#
#=     n_tokens = 5 =#
#=     seq_len = 3 =#
#=     n_repeat = 4 =#
#=     startsym = "<s>" =#
#=     endsym = "</s>" =#
#=     unksym = "<unk>" =#
#=     labels = string.(1:n_tokens) =#
#=     vocab = [unksym, startsym, endsym, labels...] =#
#=     vocab_size = length(vocab) =#
#==#
#=     n_blocks, n_heads, head_dim, hidden_dim, ff_dim = 1, 1, 1, 1, 1 =#
#=     textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym) =#
#==#
#=     attn_args = (n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim) =#
#=     block_args = (attn_args..., ff_dim=ff_dim) =#
#=     tfr_args = (block_args..., n_blocks=n_blocks, vocab_size=vocab_size) =#
#=     env_args = (n_labels = length(labels), batch_size = n_tokens ^ seq_len, seq_len = seq_len, n_repeat = n_repeat,) =#
#==#
#=     counters = default_counters() =#
#=     gene_counter = find(:type, AbstractGene, counters) =#
#=     tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),)) =#
#=     developer = Creator(TransformerPhenotype, (;textenc=textenc)) =#
#=     pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters)) =#
#=     env_creator = Creator(RepeatSequence, env_args) =#
#==#
#=     Visualizer() = create_op("Reporter", =#
#=         retriever=Jevo.PopulationRetriever(), =#
#=         operator=(s,ps)-> ( ind = ps[1][1].individuals[1]; =#
#=                            @info(string(ind.id)* " "*visualize(ind, ps[1][1])))) =#
#==#
#=     PrintOutcomeMatrix() = create_op("Reporter", =#
#=         retriever=Jevo.PopulationRetriever(), =#
#=         operator=(s,ps)-> (@info getonly(x->x isa Jevo.OutcomeMatrix, ps[1][1].data);)) =#
#==#
#=     mrs = (0.1f0, 0.01f0, 0.001f0) =#
#=     state = State("", rng, [pop_creator, env_creator],  =#
#=                   [InitializeAllPopulations(), =#
#=                     CreateMissingWorkers(1, slurm=false), =#
#=                     InitializePhylogeny(), =#
#=                     InitializeDeltaCache(), =#
#=                     UpdateParentsAcrossAllWorkers(time=true), =#
#=                     SoloMatchMaker(["p"]),  =#
#=                     Performer(time=true), =#
#=                     ComputeOutcomeMatrix(), =#
#=                     PrintOutcomeMatrix(), =#
#=                     ElitistLexicaseSelectorAndReproducer(n_inds, ϵ=true), =#
#=                     Visualizer(), =#
#=                     Mutator(mr=mrs), =#
#=                     UpdatePhylogeny(), =#
#=                     UpdateDeltaCache(), =#
#=                     ClearInteractionsAndRecords(), =#
#=                 ], counters=counters) =#
#=     run!(state, 2) =#
#= end =#
