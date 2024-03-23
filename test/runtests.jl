using Jevo
using HDF5
using Test
using StableRNGs
using Logging
using StatsBase
using Flux
using Transformers
using Transformers.TextEncoders
using Transformers.Datasets: batched

# Global variable for weight_cache
weight_cache = WeightCache(maxsize=1_000_000)

@testset "Jevo.jl" begin

rng = StableRNG(1)

@testset "counter" begin
  """Creates a counter, counts 10 times, and checks that the final value is correct."""
  c = Counter(AbstractGene)
  for i in 1:10
      inc!(c)
  end
  @test value(c) == 11
  @test find(:type, AbstractGene, [c]) == c
  try
      find(:type, AbstractIndividual, [c])
  catch
      @test true
  end

end

@testset "state" begin
  state = State()
  Jevo.operate!(state)
  @test Jevo.get_counter(AbstractGeneration, state) |> value == 2
end

@testset "HDF5Logger" begin
  rm("statistics.h5", force=true)
  with_logger(Jevo.HDF5Logger("statistics.h5")) do
      m = Measurement(GenotypeSum, 1, 1)
      sm = StatisticalMeasurement(GenotypeSum, [1,2,3], 1)
      @h5 m
      @h5 sm
  end
  h5open("statistics.h5", "r") do io
      @test haskey(io, "1/GenotypeSum/value")
      @test haskey(io, "1/GenotypeSum/min")
      @test haskey(io, "1/GenotypeSum/max")
      @test haskey(io, "1/GenotypeSum/mean")
      @test haskey(io, "1/GenotypeSum/std")
      @test haskey(io, "1/GenotypeSum/n_samples")
  end
  rm("statistics.h5", force=true)
end
@testset "JevoLogger" begin
  rm("statistics.h5", force=true)
  with_logger(Jevo.JevoLogger()) do
      sm = StatisticalMeasurement(GenotypeSum, [1,2,3], 1)
      # log to hdf5 only
      log(sm, true, false, false)
      # log to text only
      log(sm, false, true, false)
  end
  rm("statistics.h5", force=true)
end

@testset "numbers game unit and integration" begin
  # creators
  n_dims = 2
  n_inds = 2
  n_species = 2
  n_pops = 2
  counters = default_counters()
  ng_gc = ng_genotype_creator = Creator(VectorGenotype, (n=n_dims,rng=rng))
  ng_developer = Creator(VectorPhenotype)
  # genotypes
  genotype = ng_genotype_creator()
  @test length(genotype.numbers) == n_dims
  # phenotypes
  phenotype = develop(ng_developer, genotype)
  @test phenotype.numbers == genotype.numbers
  # Test that Individual has randomly generated data
  @test Individual(counters, ng_genotype_creator, ng_developer).genotype.numbers |> sum != 0
  # Population
  @test Population("ng", n_inds, ng_genotype_creator, ng_developer, counters).individuals |> length == n_inds
  # Composite population
  comp_pop = CompositePopulation("species", [("p$i", n_inds, ng_gc, ng_developer) for i in 1:n_species], counters)
  # create a composite population creator
  comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_gc, ng_developer)
                                                               for i in 1:n_species], counters))
  comp_pop = comp_pop_creator()
  @test comp_pop.populations |> length == n_species
  @test all(length(p.individuals) == n_inds for p in comp_pop.populations)
  # Create composite composite population, 2 deep binary tree
  comp_comp_pop = CompositePopulation("ecosystem",
                      [CompositePopulation("composite$i",
                          [("p$i", n_inds, ng_gc, ng_developer) for i in 1:n_species],
                      counters)
                  for i in 1:n_pops])
  # Create CompositePopulation creator
  """
      ecosystem
      ├── composite1
      │   ├── pop1: ind1a1, ind1a2
      │   └── pop2: ind1b1, ind1b2
      └── composite2
          ├── pop3: ind2a1, ind2a2
          └── pop4: ind2b1, ind2b2
  """
  pid = 0
  comp_comp_pop_creator = 
      Creator(CompositePopulation, 
          ("ecosystem", 
           [Creator(CompositePopulation,
                  ("composite$i",
                   [("p$(pid+=1)", n_inds, ng_gc, ng_developer) for p in 1:n_species],
                  counters))
          for i in 1:n_pops]))
  comp_comp_pop = comp_comp_pop_creator()
  pop_initializer = InitializeAllPopulations()
  state = State([comp_comp_pop_creator], [pop_initializer])
  @testset "PopulationCreatorRetriever" begin
      pops = [pop_creator() for pop_creator in pop_initializer.retriever(state)]
      @test length(pops) == 1
      @test pops[1].id == "ecosystem"
      run!(state, 1)
  end
  @testset "Population{Retriever,Updater}" begin
      # PopulationRetriever
      @test state.populations |> length == 1
      @test state.populations[1].id == "ecosystem"
      @test state.populations[1].populations |> length == n_pops
      pops = PopulationRetriever(["ecosystem"])(state)
      @test length(pops) == 1 # requesting ecosystem returns single population
      @test length(pops[1]) == 4
      @test typeof(pops) == Vector{Vector{Population}}
      pops = PopulationRetriever()(state) # return all pops by default
      @test length(pops) == 4
      @test length(pops[1]) == 1
      try
          PopulationRetriever(["nopop"])(state)
      catch
          @test true
      end
      pops = PopulationRetriever(["composite1"])(state)
      @test length(pops) == 1
      @test length(pops[1]) == n_species
      @test typeof(pops) == Vector{Vector{Population}}
      pops = PopulationRetriever(["p2"])(state)
      @test length(pops) == 1
      @test length.(pops) == [1] # p2 contains a single pop, p2
      pops = PopulationRetriever(["p1", "p2"])(state)
      @test length(pops) == n_pops
      @test length(pops[1]) == 1
      @test length(pops[1][1].individuals) == n_inds
  #     # find population
      @test find_population("p1", state).id == "p1"
      @test find_population("p4", state).id == "p4"
      try
          find_population("composite1", state)
      catch
          @test true
      end
      try
          find_population("nopop", state)
      catch
          @test true
      end
      # PopulationUpdater
      pops = PopulationRetriever(["p1", "p2"])(state)
      @test all([length(p) == 1 for p in pops])
      @test all([length(p[1].individuals) == 2 for p in pops])
      # bad practice, we are making individuals with the same id
      updater = PopulationUpdater(["p1"])(state, [deepcopy(pops[1][1].individuals)])
      pops = PopulationRetriever(["p1", "p2"])(state)
      @test [length(p) for p in pops] == [1,1]
      @test [length(p[1].individuals) for p in pops] == [4,2]
  end

  env_creator = Creator(CompareOnOne)
  @testset "env" begin
      env = env_creator()
  end

  # Matchmaker only
  ava = AllVsAllMatchMaker(["composite1", "composite2"])
  solo = SoloMatchMaker(["composite1", "composite2"])
  performer = Performer()
  evaluator = ScalarFitnessEvaluator(["composite1","composite2"])
  k = 1
  selector = TruncationSelector(k)
  selector_p1 = TruncationSelector(k, ["p1"])
  selector_p2_p3 = TruncationSelector(k, ["p2", "p3"])
  reproducer = UniformReproducer(n_inds)
  reproducer_p2_p3 = UniformReproducer(n_inds, ["p2", "p3"])
  mutator = Mutator()
  assertor = PopSizeAssertor(n_inds)
  ind_resetter = ClearInteractionsAndRecords()
  reporter = Reporter(GenotypeSum)

  @testset "Clone" begin
    state = State([comp_comp_pop_creator, env_creator], [pop_initializer, ava])
    run!(state, 1)
    # Clone an individual
    ind = Jevo.find_population("p1", state).individuals[1]
    clone = Jevo.clone(state, ind)
    @test clone.id != ind.id
    @test clone.generation == 2
  end

  @testset "MatchMaker" begin
      @testset "AllVsAll" begin
          state = State([comp_comp_pop_creator, env_creator], [pop_initializer, ava])
          @test Jevo.get_creators(AbstractEnvironment, state) |> length == 1
          run!(state, 1)
          n_pairs = n_pops * n_species
          n_unique_pairs = n_pairs * (n_inds^2 - n_inds) / 2
          @test length(state.matches) == n_unique_pairs * n_inds^2
      end
      @testset "Solo" begin
          state = State([comp_comp_pop_creator, env_creator], [pop_initializer, solo])
          run!(state, 1)
          @test length(state.matches) == n_inds * n_species * n_pops
      end
  end


  @testset "Performer" begin
      state = State([comp_comp_pop_creator, env_creator],
                     [pop_initializer, ava, performer])
      # Make sure no inds have interactions
      @test all(ind -> isempty(ind.interactions),
              Jevo.get_individuals(state.populations))
      run!(state, 1)
      expected_n_interactions = n_species * n_inds
      @test all(length(ind.interactions) == expected_n_interactions 
                for ind in Jevo.get_individuals(state.populations))
  end
  @testset "ScalarFitnessEvaluator" begin
      state = State([comp_comp_pop_creator, env_creator],
                     [pop_initializer, ava, performer, evaluator])
      run!(state, 1)
      @test all(ind -> length(ind.records) == 1,
              Jevo.get_individuals(state.populations))
      @test generation(state) == 2
  end
  @testset "Selector" begin
      state = State([comp_comp_pop_creator, env_creator],
                     [pop_initializer, ava, performer, evaluator, selector_p1, selector_p2_p3])
      run!(state, 1)
      @test length(Jevo.find_population("p1", state).individuals) == k
      @test length(Jevo.find_population("p2", state).individuals) == k
      @test length(Jevo.find_population("p3", state).individuals) == k
  end
  @testset "Reproducer" begin
      state = State([comp_comp_pop_creator, env_creator],
                     [pop_initializer, ava, performer, evaluator, selector_p1, selector_p2_p3, reproducer_p2_p3])
      run!(state, 1)
      @test length(Jevo.find_population("p1", state).individuals) == k
      @test length(Jevo.find_population("p2", state).individuals) == n_inds
      @test length(Jevo.find_population("p3", state).individuals) == n_inds
      @test Jevo.find_population("p3", state).individuals[1].generation == 0
      @test Jevo.find_population("p3", state).individuals[2].generation == 1
  end
  @testset "Mutator" begin
      # comp_pop_creator has n_species pops with n_inds inds
      state = State([comp_comp_pop_creator, env_creator],
                    [pop_initializer, ava, performer, evaluator, selector_p1, selector_p2_p3, reproducer_p2_p3, mutator])
      run!(state, 1)
      pops = PopulationRetriever(["p2", "p3"])(state)
      @test [length(p[1].individuals) for p in pops] == [n_inds, n_inds]
  end
  @testset "Assertor" begin
      state = State([comp_comp_pop_creator, env_creator],
                    [pop_initializer, ava, performer, evaluator, assertor, selector, reproducer, assertor, mutator, assertor])
      run!(state, 1)
      @test true
      state = State([comp_comp_pop_creator, env_creator],
                    [pop_initializer, ava, performer, evaluator, assertor, selector, assertor])
      try
          run!(state, 1)
          @test false
      catch
          @test true
      end
  end
  @testset "Reporter" begin
      rm("statistics.h5", force=true)
      rm("run.log", force=true)
      with_logger(JevoLogger()) do
          state = State([comp_comp_pop_creator, env_creator],
                        [pop_initializer, reporter])
          run!(state, 1)
      end
      rm("statistics.h5", force=true)
      rm("run.log", force=true)
  end
  @testset "Timer" begin
      with_logger(JevoLogger()) do
          state = State([comp_comp_pop_creator, env_creator],
                        [TimeReporter(AbstractGeneration), pop_initializer, TimeReporter(AbstractGeneration)])
          run!(state, 1)
      end
  end
  @testset "run multigen" begin
      println("running multigen")
      with_logger(JevoLogger()) do
          state = State([comp_comp_pop_creator, env_creator],
                        [pop_initializer, ava, performer, evaluator, assertor, selector, reproducer, assertor, mutator, reporter, ind_resetter])
          run!(state, 10)
      end
  end
  @testset "checkpointer" begin
  end
end

@testset "neural-net" begin
    @testset "unit tests" begin
        state = State()
        gene_counter = Jevo.get_counter(AbstractGene, state)
        rng = StableRNG(1)
        @test gene_counter |> value == 1
        # test creating a genotype
        relu(x) = max(0, x)
        # Naive way to create network
        Network(NoCoupling, [Jevo.Dense(Jevo.Weights((784,784),NetworkGene[]), Jevo.Weights((784,10),NetworkGene[]), relu)])
        # Better interface
        net = Network(rng, gene_counter, StrictCoupling, [(Jevo.Dense, (dims=(784,10), σ=relu))])
        dense = net.layers[1]

        @testset "binding" begin
            net_bind = Network(rng, gene_counter, StrictCoupling, [(Jevo.Dense, (dims=(784,10), σ=relu))])
            dense_bind = net_bind.layers[1]
            @test length(dense_bind.weights.muts) == 1
            seed = dense_bind.weights.muts[1].seed
            binding = Jevo.get_binding((784,10), dense_bind.weights.muts)
            @test binding.dims == (784,10)
            @test binding.last_seed == seed
            @test isnothing(binding.second_to_last_seed)
            push!(dense_bind.weights.muts, Jevo.NetworkGene(2, 2, 0.1, Jevo.apply_kaiming_normal_noise!))
            binding = Jevo.get_binding((784,10), dense_bind.weights.muts)
            @test binding.dims == (784,10)
            @test binding.last_seed == dense_bind.weights.muts[2].seed
            @test binding.second_to_last_seed == seed
        end
        @testset "tensor()" begin
            @test (10,784) == size(Jevo.tensor(dense.weights))
            @test mean(Jevo.tensor(dense.weights)) ≈ 0.0 atol=0.01
        end
        # Test constructing with weight cache
        @testset "weight cache" begin
            push!(dense.weights.muts, deepcopy(dense.weights.muts[1]))
            push!(dense.bias.muts, deepcopy(dense.bias.muts[1]))
            @test length(deepcopy(dense.weights.muts)) == 2
            @test length(deepcopy(dense.bias.muts)) == 2
            @test length(weight_cache) == 0
            nocache_construction = Jevo.create_layer(dense, weight_cache=weight_cache)
            @testset "restore from unaltered cache" begin
                @test length(weight_cache) == 2
                # Test layer construction using cache and confirm the results are the same
                cache_construction = Jevo.create_layer(dense, weight_cache=weight_cache)
                @test length(weight_cache) == 2
                @test nocache_construction.weight == cache_construction.weight
                @test nocache_construction.bias == cache_construction.bias
            end
            @testset "restore from altered cache" begin
                # Modify weight cache, then confirm that the results are different
                for arr in values(weight_cache)
                    arr .-= 999
                end
                # Add a mutation to the gene to restore from a parent 
                push!(dense.weights.muts, Jevo.NetworkGene(3, 3, 0.1, Jevo.apply_kaiming_normal_noise!))
                cache_construction = Jevo.create_layer(dense, weight_cache=weight_cache)
                @test nocache_construction.weight != cache_construction.weight
                @test all(cache_construction.weight .< -900)
            end
        end
        # Test phenotype creation & forward pass
        @testset "develop & forward pass full rank" begin
            net = Network(rng, gene_counter, StrictCoupling, [(Jevo.Dense, (dims=(784,10), σ=relu))])
            dense = net.layers[1]
            creator = Creator(Model)
            model = develop(creator, net)
            @test model |> typeof <: Model 
            @test rand(Float32, 784) |> model.chain |> size == (10,)
            # confirm we can get a list of weights 
            @test length(Jevo.get_weights(rng, net, n=-1)) == 2
            @test length(Jevo.get_weights(rng, net, n=1)) == 1
            # Add mutations to each network
            mutated_net = Jevo.mutate(rng, state, net, mr=Float32(0.01))
            @test all(map(w ->length(w.muts)==2, Jevo.get_weights(rng, mutated_net, n=-1)))
            mutated_net = Jevo.mutate(rng, state, mutated_net, mr=Float32(0.01))
            @test all(map(w ->length(w.muts)==3, Jevo.get_weights(rng, mutated_net, n=-1)))
        end
        @testset "low rank develop + fwd" begin
            creator = Creator(Model)
            full_model = develop(creator, Network(rng, gene_counter, StrictCoupling, [(Jevo.Dense, (dims=(784,10), σ=relu))]))
            recon_full_model = develop(creator, Network(rng, gene_counter, StrictCoupling, [(Jevo.Dense, (dims=(784,10), σ=relu, rank=10))]))
            lora_model = develop(creator, Network(rng, gene_counter, StrictCoupling, [(Jevo.Dense, (dims=(784,10), σ=relu, rank=1))]))

            @test rand(Float32, 784) |> full_model.chain |> size == (10,)
            @test rand(Float32, 784) |> recon_full_model.chain |> size == (10,)
            @test rand(Float32, 784) |> lora_model.chain |> size == (10,)

            dense = full_model.chain.layers[1]
            f_m, f_std = mean(dense.weight), std(dense.weight)
            dense = recon_full_model.chain.layers[1]
            r_m, r_std = mean(dense.weight), std(dense.weight)
            dense = lora_model.chain.layers[1]
            lora_m, lora_std = mean(dense.weight), std(dense.weight)

            @test r_m ≈ f_m atol=0.01
            @test r_std ≈ f_std atol=0.01
            @test lora_m ≈ f_m atol=0.01
            @test !isapprox(lora_std, f_std, atol=0.01)

        end
        @testset "Transformer" begin
            state = State()
            gene_counter = Jevo.get_counter(AbstractGene, state)
            n_blocks = 2
            n_heads = 2
            head_dim = 5
            hidden_dim = 10
            ff_dim = 20
            startsym = "<s>"
            endsym = "</s>"
            unksym = "<unk>"
            labels = string.(1:5)
            vocab = [unksym, startsym, endsym, labels...]
            vocab_size = length(vocab)

            attn_args = (n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim)
            block_args = (attn_args..., ff_dim=ff_dim)
            tfr_args = (block_args..., n_blocks=n_blocks, vocab_size=vocab_size)
            # Construct each of the pieces of a transformer
            # Embed, EmbedDecoder, SelfAttention, PostNormResidual, TransformerDecoderBlock, Transformer
            rng = StableRNG(1)
            embed, embed_decoder = Jevo.create_embeds(rng, gene_counter, (vocab_size, head_dim))
            sa = Jevo.SelfAttention(rng, gene_counter; attn_args...) 
            pnr_sa = Jevo.PostNormResidual(rng, gene_counter, sa; hidden_dim=hidden_dim)
            db = Jevo.TransformerDecoderBlock(rng, gene_counter; block_args...)
            trf = Jevo.Transformer(rng, gene_counter; tfr_args...)
            net = Network(rng, gene_counter, StrictCoupling, [(Jevo.Transformer, tfr_args)])
            weights = Jevo.get_weights(rng, net, n=-1)
            dims = [w.dims for w in weights]
            @test (hidden_dim,vocab_size) in dims # embed
            @test (vocab_size,) in dims # embed bias
            # make sure (10,5) is in dims exactly once, we don't want to
            # double count when sharing weights for embed/embeddecoder
            @test sum([d == (hidden_dim,vocab_size) for d in dims]) == 1
            @test (hidden_dim,ff_dim) in dims # ff
            @test (ff_dim,hidden_dim) in dims
            @test (3*head_dim*n_heads, hidden_dim) in dims # qkv
            @test (head_dim*n_heads, hidden_dim) in dims   # out
            @test (hidden_dim,) in dims # layernorm
            mutated_net = Jevo.mutate(rng, state, net, mr=Float32(0.01))
            # TODO ADD TEST FOR GAUSSIAN VS KAIMING INIT
            Jevo.create_layer(embed; weight_cache=weight_cache)
            Jevo.create_layer(embed_decoder; weight_cache=weight_cache)
            Jevo.create_layer(sa; weight_cache=weight_cache)
            Jevo.create_layer(pnr_sa; weight_cache=weight_cache)
            Jevo.create_layer(db; weight_cache=weight_cache)
            Jevo.create_layer((db,db); weight_cache=weight_cache)

            textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)
            creator = Creator(Jevo.TransformerPhenotype, (;textenc=textenc))
            trf_p = develop(creator, net)
            seq = "1 2 1"
            input = preprocess(trf_p, seq)
            logits = trf_p(input)
            @test size(logits) == (8, 5, 1)
            # batching & masking
            sample_batch = batched([(seq,) for i in 1:100])[1]
            input_batch = preprocess(trf_p, sample_batch)
            logits = trf_p(input_batch)
            @test size(logits) == (8, 5, 100)
            env = RepeatSequence(vocab_size=vocab_size,
                                 seq_len=8,
                                 batch_size=7,
                                 n_repeat=3)
            @test length(Jevo.step!(env, [trf_p])) == 1
            @test length(Jevo.play(env, [trf_p])) == 1
            seq, logits = infer(trf_p, "1 2 1")
            # TODO TEST EXTENSIVELY
            @testset "LowRank" begin
                # LowRank
                net = Network(rng, gene_counter, StrictCoupling, [(Jevo.Dense, (dims=(784,10), σ=relu, rank=2))])
                dense = net.layers[1]
                d = Jevo.create_layer(dense; weight_cache=weight_cache)
                @test d.weight |> size == (10, 784)
                @test d.weight |> typeof == Array{Float32,2}
                @test d.bias |> size == (10,)
                @test d.bias |> typeof == Array{Float32,1}
                lora_tfr_args = (tfr_args..., qkv_rank=2, o_rank=2, ff_rank=2, embed_rank=2)
                net = Network(rng, gene_counter, StrictCoupling, [(Jevo.Transformer, lora_tfr_args)])
                lora_tfr_p = develop(Creator(Jevo.TransformerPhenotype, (;textenc=textenc)), net)
            end
        end
    end
    @testset "integration tests" begin
        n_env_inputs = 5
        n_species = 2
        n_inds = 3
        empty!(weight_cache)
    
        state = State()
        gene_counter = Jevo.get_counter(AbstractGene, state)
        geno_creator = Creator(Network, (rng, gene_counter, StrictCoupling, [(Jevo.Dense, (dims=(n_env_inputs,1), σ=relu))]))
        geno = geno_creator()
        phen_creator = Creator(Model)
        phen = develop(phen_creator, geno)
        net = Network(rng, gene_counter, StrictCoupling, [(Jevo.Dense, (dims=(n_env_inputs,1), σ=relu))])
        comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, geno_creator, phen_creator) for i in 1:n_species], state.counters))
        comp_pop = comp_pop_creator()
        @test comp_pop.populations |> length == n_species
        @test all(length(p.individuals) == n_inds for p in comp_pop.populations)
        env_creator = Creator(MaxLogits, (n=n_env_inputs,))
        state = State([comp_pop_creator, env_creator], 
              [ InitializeAllPopulations(), 
                SoloMatchMaker(["p1", "p2"]), 
                Performer(),
                ScalarFitnessEvaluator(["p1", "p2"]), 
                TruncationSelector(1),
                UniformReproducer(n_inds),
                Mutator(;mr=Float32(0.01), n=2),
                PopSizeAssertor(n_inds),
                ClearInteractionsAndRecords()])
        @test length(state.matches) == 0
        run!(state, 1)
    end
    # @testset "mnist" begin
    #     n_env_inputs = 784
    #     n_species = 1
    #     n_inds = 3
    #     empty!(weight_cache)
    # 
    #     state = State()
    #     gene_counter = Jevo.get_counter(AbstractGene, state)
    #     geno_creator = Creator(Network, (rng, gene_counter, StrictCoupling, [(Jevo.Dense, (n_env_inputs,10), relu)]))
    #     geno = geno_creator()
    #     phen_creator = Creator(Model)
    #     phen = develop(phen_creator, geno)
    #     net = Network(rng, gene_counter, StrictCoupling, [(Jevo.Dense, (n_env_inputs,10), relu)])
    #     comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, geno_creator, phen_creator) for i in 1:n_species], state.counters))
    #     comp_pop = comp_pop_creator()
    #     env_creator = Creator(MNISTEnv)
    #     state = State([comp_pop_creator, env_creator], 
    #           [ InitializeAllPopulations(), 
    #             SoloMatchMaker(["p1"]), 
    #             Performer(),
    #             ScalarFitnessEvaluator(["p1"]), 
    #             TruncationSelector(1),
    #             UniformReproducer(n_inds),
    #             Mutator(;mr=Float32(0.01), n=2),
    #             PopSizeAssertor(n_inds),
    #             ClearInteractionsAndRecords()])
    #     run!(state, 1)
    # end
    @testset "nn confirm improvement" begin
        # TODO
    end
    @testset "distributed eval" begin
        # TODO
    end
end

end
