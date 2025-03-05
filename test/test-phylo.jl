using Jevo: Interaction, EstimatedInteraction, PhylogeneticTree, add_child!, RandomlySampledInteractions, find_k_nearest_interactions, weighted_average_outcome, estimate!, RelatedOutcome
using DataStructures: SortedDict

@testset "phylo" begin

n_dims = 2
n_inds = 10
n_species = 2
n_pops = 2
counters = default_counters()
ng_gc = ng_genotype_creator = Creator(VectorGenotype, (n=n_dims,rng=rng))
ng_developer = Creator(VectorPhenotype)
env_creator = Creator(CompareOnOne)

pop_initializer = InitializeAllPopulations()

comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_gc, ng_developer)
                        for i in 1:n_species], counters))
k, n_inds, n_gens = 2, 10, 10

@testset "PhylogenyOnly" begin
    # Test initialization and updates of phylogeny
    state = State("", rng, [comp_pop_creator], 
                  [pop_initializer,
                   InitializePhylogeny(),
                   RandomEvaluator(),
                   TruncationSelector(k),
                   CloneUniformReproducer(n_inds),
                   UpdatePhylogeny(),
                   PurgePhylogeny(),
                   ClearInteractionsAndRecords(),
                  ], counters=counters)
    run!(state, 1)

    for p in state.populations
        for subpop in p.populations
            tree = Jevo.get_tree(subpop)
            @test length(tree.genesis) == k
        end
    end

    run!(state, n_gens)
    for p in state.populations
        for subpop in p.populations
            tree = Jevo.get_tree(subpop)
            genesis_ids = Set([i.id for i in tree.genesis])
            @test length(tree.genesis) == 1
            @test n_inds < length(tree.tree) < n_inds + ((n_inds-k)*(n_gens))
            for node in values(tree.tree)
                @test !isnothing(node.id) || node.id ∈ genesis_ids
            end
        end
    end
end

@testset "Estimate" begin

    @testset "WeightedAverage" begin
        # Only the last three arguments matter for a RelatedOutcome:
        #   distance, outcome_a, outcome_b
        # two interactions which are equally close to the target interaction
        @assert weighted_average_outcome([RelatedOutcome(0,0,1,1,0),
                                          RelatedOutcome(0,0,1,0,1)]) == (0.5, 0.5)
        # Test one relative
        @assert weighted_average_outcome([RelatedOutcome(0,0,1,1,0)]) == (1, 0)
    
        # Test one close relative and one far relative
        @assert weighted_average_outcome([RelatedOutcome(0,0,1,1,0),
                                          RelatedOutcome(1,0,9,0,1)]) == (0.9, 0.1)
        # Test two close relatives and one far relative
        @assert weighted_average_outcome([RelatedOutcome(0,0,1,1,0),
                                          RelatedOutcome(0,0,1,1,0),
                                          RelatedOutcome(0,0,8,0,1)]) == (0.9, 0.1)
        # Test four far relatives
        @assert weighted_average_outcome([RelatedOutcome(0,0,8,1,0),
                                          RelatedOutcome(0,0,8,1,0),
                                          RelatedOutcome(0,0,8,1,0),
                                          RelatedOutcome(0,0,8,0,1)]) == (0.75, 0.25)
    end

    @testset "find_k_nearest_interactions" begin
        @testset "6,14" begin
        #   A             B
        #-------        ------
        #   1            10
        #   |             |
        #   2            11
        #  / \           / \
        # 3   4        12   13
        # |   |         |   |
        # 5   6        14   15
        treeA = PhylogeneticTree([1])
        add_child!(treeA, 1, 2)
        add_child!(treeA, 2, 3)
        add_child!(treeA, 2, 4)
        add_child!(treeA, 3, 5)
        add_child!(treeA, 4, 6)
        treeB = PhylogeneticTree([10])
        add_child!(treeB, 10, 11)
        add_child!(treeB, 11, 12)
        add_child!(treeB, 11, 13)
        add_child!(treeB, 12, 14)
        add_child!(treeB, 13, 15)
        
        # make empty interaction outcomes dict with all individuals in both trees
        io = Dict{Int, SortedDict{Int, Float64}}(i=>SortedDict{Int, Float64}() for i in 1:15)
        
        # first two interactions are close to (6,14)
        io[4][14],io[14][4] = 1, 0 
        io[6][12],io[12][6] = 1, 0
        # these two are far away
        io[1][10],io[10][1] = 0, 1
        io[2][11],io[11][2] = 0, 1
    
        expected_dists_from_6_14 = [1, 1, 4, 6]
        max_dist = maximum(expected_dists_from_6_14)
        for k in 1:4
            nearest = find_k_nearest_interactions(6, 14, treeA, treeB, io, k, max_dist=max_dist)
            @test length(nearest) == k
            @test [n.dist for n in nearest] == expected_dists_from_6_14[1:k]
        end
        # Test that it works when the trees are swapped
        for k in 1:4
            nearest = find_k_nearest_interactions(14, 6, treeB, treeA, io, k, max_dist=max_dist)
            @test length(nearest) == k
            @test [n.dist for n in nearest] == expected_dists_from_6_14[1:k]
        end
    
        end
        @testset "Disconnected" begin
        # Test that we don't find any interactions that are not reachable
        #  A            B
        #-----        -----
        # 1 2         3  4
        # | |         |  |
        # 5 6         7  8
        treeA = PhylogeneticTree([1, 2])
        add_child!(treeA, 1, 5)
        add_child!(treeA, 2, 6)
        treeB = PhylogeneticTree([3, 4])
        add_child!(treeB, 3, 7)
        add_child!(treeB, 4, 8)
        io = Dict{Int, SortedDict{Int, Float64}}(i=>SortedDict{Int, Float64}() for i in 1:8)
    
        io[1][3],io[3][1] = 1, 0
        io[2][4],io[4][2] = 1, 0
        io[2][8],io[8][2] = 1, 0
    
        k = 2
        nearest = find_k_nearest_interactions(5, 7, treeA, treeB, io, k, max_dist=5)
        @test length(nearest) == 1 # This raises a warning, but it's what we expect
        @test nearest[1] == RelatedOutcome(1, 3, 2, 1, 0)
    
        k = 3
        nearest = find_k_nearest_interactions(6, 8, treeA, treeB, io, k, max_dist=5)
        @test length(nearest) == 2 # This raises a warning, but it's what we expect
        @test nearest[1] == RelatedOutcome(2, 8, 1, 1, 0)
        @test nearest[2] == RelatedOutcome(2, 4, 2, 1, 0)
        end
    end


    @testset "estimate!" begin

        # popa
        # 1 2 3
        # | | |
        # 4 5 6
        s = State()

        inds = [Individual(i, 0, Int[], ng_gc(), ng_developer) for i in 1:3]
        pop = Population("p1", inds)
        Jevo.initialize_phylogeny!(s, pop) 

        push!(pop.data, Jevo.RandomlySampledInteractions("p1", [(4,4), (5,5)]))

        for i in 4:6
            push!(pop.individuals, Individual(i, 1, [i-3], ng_gc(), ng_developer))
        end

        Jevo.update_phylogeny!(s, pop)

        ind = pop.individuals
        ind[1].interactions = [ Interaction(1, [1], 1.0), Interaction(1, [2], 1.0), Interaction(1, [3], 1.0), Interaction(1, [4], 1.0), Interaction(1, [5], 1.0) ]
        ind[2].interactions = [ Interaction(2, [1], 1.0), Interaction(2, [2], 1.0), Interaction(2, [3], 1.0), Interaction(2, [4], 1.0), Interaction(2, [5], 1.0) ]
        ind[3].interactions = [ Interaction(3, [1], 1.0), Interaction(3, [2], 1.0), Interaction(3, [3], 1.0), Interaction(3, [4], 1.0), Interaction(3, [5], 1.0) ]
        ind[4].interactions = [ Interaction(4, [1], 1.0), Interaction(4, [2], 0.0), Interaction(4, [3], 1.0), Interaction(4, [4], 1.0) ]
        ind[5].interactions = [ Interaction(5, [1], 0.0), Interaction(5, [2], 1.0), Interaction(5, [3], 1.0), Interaction(5, [5], 1.0) ]
        ind[6].interactions = [ Interaction(6, [1], 1.0), Interaction(6, [2], 1.0), Interaction(6, [3], 1.0) ]

        k, max_dist=2, 10

        Jevo.estimate!(s, [pop, pop], k, max_dist)

        estimate_4_5 =filter(i->i isa EstimatedInteraction && i.other_ids[1] == 5, ind[4].interactions)
        estimate_5_4 =filter(i->i isa EstimatedInteraction && i.other_ids[1] == 4, ind[5].interactions)
        @test length(estimate_4_5) == length(estimate_5_4) == 1
        @test estimate_4_5[1].score == 0.5
        @test estimate_5_4[1].score == 0.5

        Jevo.add_outcome_matrices!(s, [pop])

        outcome_matrix = getonly(x->x isa Jevo.OutcomeMatrix, pop.data)
        @test size(outcome_matrix.matrix) == (6, 6)
        @test outcome_matrix.matrix[1,1] == 1.0 # non-estimate
        @test outcome_matrix.matrix[1,2] == 1.0 # non-estimate
        @test outcome_matrix.matrix[4,5] == 0.5 == outcome_matrix.matrix[5,4] # estimate
        @test outcome_matrix.matrix[6,6] == 1.0 # estimated from 3,3

        @test 1 ∈ keys(getonly(x->x isa Jevo.OutcomeCache, s.data).cache)
        @test 1 ∈ keys(getonly(x->x isa Jevo.OutcomeCache, s.data).cache[1])
        @test 4 ∈ keys(getonly(x->x isa Jevo.OutcomeCache, s.data).cache)
        @test 4 ∈ keys(getonly(x->x isa Jevo.OutcomeCache, s.data).cache[4])
        @test 5 ∉ keys(getonly(x->x isa Jevo.OutcomeCache, s.data).cache[4]) # estimated

    end
    @testset "old_vs_new_matchmaker" begin
        s = State()

        inds = [Individual(i, 0, Int[], ng_gc(), ng_developer) for i in 1:3]
        pop = Population("p1", inds)
        Jevo.initialize_phylogeny!(s, pop) 

        for i in 4:6
            push!(pop.individuals, Individual(i, 1, [i-3], ng_gc(), ng_developer))
        end

        
        @testset "no_cached_matches" begin
            no_cached_matches = true
            matches = Jevo.make_old_vs_new_matches(s, [[pop]], no_cached_matches, env_creator=env_creator)
            @test length(matches) == 18
            match_ids = [Tuple(ind.id for ind in match.individuals) for match in matches]
            for i in 1:3, j in 4:6
                @test (i, j) in match_ids
                @test (j, i) in match_ids
            end
        end

        outcome_cache = Jevo.OutcomeCache(["p1"], Jevo.LRU{Int, Dict{Int, Float64}}(maxsize=100))
        for i in 1:6
            outcome_cache.cache[i] = Dict{Int, Float64}()
            for j in 1:6
                outcome_cache.cache[i][j] = 1.0
            end
        end
        push!(s.data, outcome_cache)

        @testset "cached_matches" begin
            no_cached_matches = false
            matches = Jevo.make_old_vs_new_matches(s, [[pop]], no_cached_matches, env_creator=env_creator)
            @test length(matches) == 0
        end

        @testset "restore_cached_outcomes" begin
            Jevo.restore_cached_outcomes!(s, [[pop]])
            for ind in pop.individuals
                @test length(ind.interactions) == 6
                all(i.score == 1.0 for i in ind.interactions)
            end
        end
    end
end

end
