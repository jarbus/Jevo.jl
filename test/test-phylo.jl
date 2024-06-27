@testset "phylo" begin

n_dims = 2
n_inds = 10
n_species = 2
n_pops = 2
counters = default_counters()
ng_gc = ng_genotype_creator = Creator(VectorGenotype, (n=n_dims,rng=rng))
ng_developer = Creator(VectorPhenotype)

pop_initializer = InitializeAllPopulations()

comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_gc, ng_developer)
                        for i in 1:n_species], counters))
@testset "PhylogenyOnly" begin
    # Test initialization and updates of phylogeny
    k = 2
    n_inds = 10
    n_gens = 10
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


@testset "DeltaCache+Genepool+Mutator" begin
end
end
