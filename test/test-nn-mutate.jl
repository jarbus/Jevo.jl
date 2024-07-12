@testset "mutation tests" begin
    # Here, we test the following:
    # 1. That with no phylogeny, gene pool, etc, we can still add mutations using a Mutator
    # 2. That with phylogeny, deltas, and a gene pool, we can add mutations with a:
    #     * ClearCurrentGenWeights
    #     * Mutator
    #     * NNGenePoolMutator
    #     * NNGenePoolReseedMutator

    n_inds = 3
    k=2
    n_tokens = 5
    startsym = "<s>"
    endsym = "</s>"
    unksym = "<unk>"
    labels = string.(0:n_tokens-1)
    vocab = [unksym, startsym, endsym, labels...]
    vocab_size = length(vocab)

    n_blocks, n_heads, head_dim, hidden_dim, ff_dim = 3, 2, 5, 10, 20
    textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)

    attn_args = (n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim)
    block_args = (attn_args..., ff_dim=ff_dim)
    tfr_args = (block_args..., n_blocks=n_blocks, vocab_size=vocab_size)
    env_args = (vocab_size = vocab_size, batch_size = 2, seq_len = 3, n_repeat = 2,)
    env_creator = Creator(RepeatSequence, env_args)
    mrs = (0.1f0, 0.01f0)

    @testset "1. Plain NN Mutator" begin
        counters = default_counters()
        gene_counter = find(:type, AbstractGene, counters)
        tfr_gc = Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)]))
        developer = Creator(TransformerPhenotype, (;textenc=textenc))
        pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
        state = State("", rng, [pop_creator, env_creator], 
                      [InitializeAllPopulations(),
                        RandomEvaluator(["p"]), 
                        TruncationSelector(k),
                        CloneUniformReproducer(n_inds),
                        Mutator(mr=mrs),
                        ClearInteractionsAndRecords(),
                    ], counters=counters)
        run!(state, 10)
        for ind in state.populations[1].individuals
            more_than_one_mutation = map(ind.genotype, weights_only=true) do layers
                weight = layers[end]
                # Check first weight has mr 1, initialization
                @assert length(weight.muts) > 0
                @test weight.muts[1].mr == 1f0
                # Check all other weights fall into mr
                for m in weight.muts[2:end]
                    @test m.mr == 0.1f0 || m.mr == 0.01f0
                end
                return length(weight.muts) > 1
            end |> any # confirm at least one weight has multiple mutations
            @test more_than_one_mutation
        end
    end
    @testset "2. ClearCurrentGenWeights" begin
        counters = default_counters()
        gene_counter = find(:type, AbstractGene, counters)
        tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
        developer = Creator(TransformerPhenotype, (;textenc=textenc))
        pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
        state = State("", rng, [pop_creator, env_creator], 
                      [InitializeAllPopulations(),
                        InitializePhylogeny(),
                        InitializeDeltaCache(),
                        RandomEvaluator(["p"]), 
                        TruncationSelector(k),
                        CloneUniformReproducer(n_inds),
                        ClearCurrentGenWeights(),
                        UpdatePhylogeny(),
                        UpdateDeltaCache(),
                        ClearInteractionsAndRecords(),
                    ], counters=counters)
        run!(state, 10)
        for ind in state.populations[1].individuals
            @test all(map(layers->length(layers[end].muts) == 0, ind.genotype, weights_only=true))
        end
    end
    @testset "3. ClearCurrentGenWeights + Mutator" begin
        counters = default_counters()
        gene_counter = find(:type, AbstractGene, counters)
        tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
        developer = Creator(TransformerPhenotype, (;textenc=textenc))
        pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
        state = State("", rng, [pop_creator, env_creator], 
                      [InitializeAllPopulations(),
                        InitializePhylogeny(),
                        InitializeDeltaCache(),
                        RandomEvaluator(["p"]), 
                        TruncationSelector(k),
                        CloneUniformReproducer(n_inds),
                        ClearCurrentGenWeights(),
                        Mutator(mr=mrs),
                        UpdatePhylogeny(),
                        UpdateDeltaCache(),
                        ClearInteractionsAndRecords(),
                    ], counters=counters)
        run!(state, 10)
        for ind in state.populations[1].individuals
            @test all(map(layers->length(layers[end].muts)  < 2, ind.genotype, weights_only=true))
            @test any(map(layers->length(layers[end].muts) == 1, ind.genotype, weights_only=true))
        end
    end
    @testset "4. NNGenePoolMutator" begin
        counters = default_counters()
        gene_counter = find(:type, AbstractGene, counters)
        tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
        developer = Creator(TransformerPhenotype, (;textenc=textenc))
        pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
        state = State("", rng, [pop_creator, env_creator], 
                      [InitializeAllPopulations(),
                        InitializePhylogeny(),
                        InitializeDeltaCache(),
                        RandomEvaluator(["p"]), 
                        TruncationSelector(k),
                        UpdateGenePool(after_gen=1, n_latest=1),
                        CloneUniformReproducer(n_inds),
                        ClearCurrentGenWeights(),
                        Mutator(mr=mrs, condition=first_gen),
                        NNGenePoolReseedMutator(["p"]; mr=mrs, prob=0.50, condition=s->generation(s)>1),
                        UpdatePhylogeny(),
                        UpdateDeltaCache(),
                        ClearInteractionsAndRecords(),
                    ], counters=counters)
        run!(state, 3)
        for ind in state.populations[1].individuals
            @test all(map(layers->length(layers[end].muts)  < 2, ind.genotype, weights_only=true))
            @test any(map(layers->length(layers[end].muts) == 1, ind.genotype, weights_only=true))
        end
    end
end
