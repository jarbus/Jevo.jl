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
# Test initialization of phylogeny
state = State(rng, [comp_pop_creator], [pop_initializer, InitializePhylogeny()])
run!(state, 10)
# Test that there is a phylogenetic tree added
for p in state.populations
    for subpop in p.populations
        tree = Jevo.get_tree(subpop)
        @test tree isa PhylogeneticTree
        @test length(tree.genesis) == length(tree.tree) == n_inds
    end
end
# Test updating phylogeny
k = 2
n_gens = 10

# need to reset the counters for this bad boy
counters = default_counters()

comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_gc, ng_developer)
                                                             for i in 1:n_species], counters))
state = State("", rng, [comp_pop_creator], 
              [pop_initializer,
               InitializePhylogeny(),
               RandomEvaluator(),
               TruncationSelector(k),
               CloneUniformReproducer(n_inds),
               UpdatePhylogeny(),
               ClearInteractionsAndRecords(),
              ], counters=counters)
run!(state, n_gens)

for p in state.populations
    for subpop in p.populations
        tree = Jevo.get_tree(subpop)
        @test n_inds < length(tree.tree) < n_inds + ((n_inds-k)*(n_gens))
    end
end


@testset "DeltaCache+Genepool+Mutator" begin
end
end
