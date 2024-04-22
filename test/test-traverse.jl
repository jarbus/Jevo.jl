state = State()
rng = StableRNG(1)
gene_counter = Jevo.get_counter(AbstractGene, state)
n_blocks, n_heads, head_dim, hidden_dim, ff_dim, vocab_size = 2, 2, 5, 10, 20, 5

attn_args = (n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim)
block_args = (attn_args..., ff_dim=ff_dim)
tfr_args = (block_args..., n_blocks=n_blocks, vocab_size=vocab_size)

@testset "HierarchicalTransformerTraverse" begin
    net = Network(rng, gene_counter, [(Jevo.Transformer, tfr_args)])
    # Check that we can use map to get all the weights of a network, should probably confirm
    # that ALL weights are retrieved, but that's probably another 20+ mins
    function get_n_muts(net)
        map(net, weights_only=true) do layers
            length(layers[end].muts)
        end
    end
    n_muts = get_n_muts(net)
    @test length(n_muts) > 0 && all(n_muts .== 1)
    # Check that we can modify all the weights using map!
    map!(net, weights_only=true) do layers
        push!(layers[end].muts, NetworkGene(0,0,0,zero))
    end
    n_muts = get_n_muts(net)
    @test length(n_muts) > 0 && all(n_muts .== 2)
    mutated_net = mutate(rng, state, net, mr=0.1f0)
    n_muts = get_n_muts(mutated_net)
end

@testset "Transformer Delta+Reproduce+Mutate" begin

k = 1
n_inds = 2
developer = Creator(Model)
tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
counters = default_counters()
parent_geno = tfr_gc()
parent_weights = Jevo.get_weights(parent_geno, no_layer_norm=true)
n_mutation_targets = length(Jevo.get_weights(parent_geno, no_layer_norm=true))

@testset "Mutation" begin
    # Test mutation for different numbers of mutations
    for n in (1, 2, -1)
        child_geno = mutate(rng, state, parent_geno, mr=0.1f0, n=n)
        child_weights = Jevo.get_weights(child_geno, no_layer_norm=true)
        @test length(parent_weights) == length(child_weights)
        n_muts = 0  # number of child mutations discovered
        for (p_w, c_w) in zip(parent_weights, child_weights)
            if length(p_w.muts) == length(c_w.muts) == 1
                @test p_w.muts[1].id   != c_w.muts[1].id
                @test p_w.muts[1].seed != c_w.muts[1].seed
                n_muts += 1
            end
        end
        if n == -1 
            @test n_muts == n_mutation_targets 
        else
            @test n_muts == n
        end
    end
end

@testset "Integrate mutation + reproduction" begin
    # Integration test
    pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
    state = State("", rng, [pop_creator], 
                  [InitializeAllPopulations(),
                   InitializePhylogeny(),
                   InitializeDeltaCache(),
                   RandomEvaluator(),
                   TruncationSelector(k),
                   CloneUniformReproducer(n_inds),
                   Mutator(;mr=Float32(0.01), n=-1),
                   UpdatePhylogeny(),
                   UpdateDeltaCache(),
                   ClearInteractionsAndRecords(),
                  ], counters=counters)
    run!(state, 1)
    parents = state.populations[1].individuals |> x->filter(i->i.generation==0, x)
    children = state.populations[1].individuals |> x->filter(i->i.generation==1, x)
    @test length(parents) == k
    @test length(children) == n_inds - k
    # test that each parent has a mutation of MR=1 and each child has a mutation of MR=0.01
    dc = Jevo.get_delta_cache(state.populations[1])
    for parent in parents, w in Jevo.get_weights(parent.genotype, no_layer_norm=true)
        @test length(w.muts) == 1
        @test w.muts[1].mr == 1
        @test parent.id ∈ keys(dc)
    end
    for child in children, w in Jevo.get_weights(child.genotype, no_layer_norm=true)
        @test length(w.muts) == 1
        @test w.muts[1].mr == 0.01f0
        @test child.id ∈ keys(dc)
    end
end

@testset "Distribute and Develop" begin
end
# @testset "Develop" begin
# end

end
