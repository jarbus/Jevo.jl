@testset "mutation tests" begin
  # Here, we test the following:
  # 1. That with no phylogeny, gene pool, etc, we can still add mutations using a Mutator
  # 2. That with phylogeny, deltas, and a gene pool, we can add mutations with a:
  #     * ClearCurrentGenWeights
  #     * Mutator
  #     * NNGenePoolMutator
  #     * NNGenePoolReseedMutator

  n_inds = 4
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
  # need new counters etc for each state
  function delta_counters_and_pop_creator()
      counters = default_counters()
      gene_counter = find(:type, AbstractGene, counters)
      tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
      developer = Creator(TransformerPhenotype, (;textenc=textenc))
      pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
      return counters, pop_creator
  end

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
      counters, pop_creator = delta_counters_and_pop_creator()
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
      run!(state, 1)
      # confirm elites from first gen still have muts
      for ind in state.populations[1].individuals[1:2]
          @test any(map(layers->length(layers[end].muts) == 1, ind.genotype, weights_only=true))
      end
      # confirm non-elites have no muts
      for ind in state.populations[1].individuals[3:4]
          @test all(map(layers->length(layers[end].muts) == 0, ind.genotype, weights_only=true))
      end
      run!(state, 10)
      # confirm all descendants have no muts 
      for ind in state.populations[1].individuals
          @test all(map(layers->length(layers[end].muts) == 0, ind.genotype, weights_only=true))
      end
  end
  @testset "3. ClearCurrentGenWeights + Mutator" begin
      counters, pop_creator = delta_counters_and_pop_creator()
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
    counters, pop_creator = delta_counters_and_pop_creator()
    state = State("", rng, [pop_creator, env_creator], 
                  [InitializeAllPopulations(),
                    InitializePhylogeny(),
                    InitializeDeltaCache(),
                    RandomEvaluator(["p"]), 
                    TruncationSelector(k),
                    UpdateGenePool(after_gen=5, n_latest=4),
                    ComputeGenePoolNetwork(after_gen=5),
                    CloneUniformReproducer(n_inds),
                    ClearCurrentGenWeights(),
                    Mutator(mr=mrs, condition=s->generation(s)<=5),
                    NNGenePoolMutator(["p"]; mr=mrs, condition=s->generation(s)>5),
                    UpdatePhylogeny(),
                    UpdateDeltaCache(),
                    ClearInteractionsAndRecords(),
                ], counters=counters)
    run!(state, 10)
    sparse_init = false
    full_init   = false
    seen_seeds  = Set()
    reused_seed = false
    for pop in state.populations
      # Confirm all new individuals have at least one gene pool mutations
      # Should have at least one sparse and one full init
      # There should also be a re-used seed in the delta cache
      for ind in pop.individuals
        @test all(map(layers->length(layers[end].muts)  < 2, ind.genotype, weights_only=true))
        @test any(map(layers->length(layers[end].muts) == 1, ind.genotype, weights_only=true))
        ind.generation != generation(state) - 1 && continue # Only check new individuals
        gp = Jevo.getonly(x->x isa GenePool, pop.data)
        g_ws = get_weights(ind.genotype, no_layer_norm=true)

        for w in g_ws
            length(w.muts) == 0 && continue
            full_init = full_init || w.muts[1].init! in (Jevo.apply_kaiming_normal_noise!,
                                                         Jevo.apply_kaiming_normal_noise_factored!)
            sparse_init = sparse_init || w.muts[1].init! == Jevo.apply_sparse_noise!
        end
      end
      for d in getonly(x->x isa Jevo.DeltaCache, pop.data) |> values
        for d_w in get_weights(d, no_layer_norm=true)
          length(d_w.muts) == 0 && continue
          reused_seed = reused_seed || d_w.muts[1].seed in seen_seeds
          push!(seen_seeds, d_w.muts[1].seed)
        end
      end
    end
    @test sparse_init
    @test full_init
    @test reused_seed
  end
end
