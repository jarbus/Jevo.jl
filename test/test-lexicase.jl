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
@testset "numbers game lexicase" begin
  n_dims = 2
  n_inds = 100
  n_species = 2
  counters = default_counters()
  ng_gc = Creator(VectorGenotype, (n=n_dims,rng=rng))
  ng_developer = Creator(VectorPhenotype)
  comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_gc, ng_developer) for i in 1:n_species], counters))
  env_creator = Creator(CompareOnOne)
  isfile("statistics.h5") && rm("statistics.h5")
  min_sum_computer = create_op("Reporter",
                        operator=(s,_) -> measure(GenotypeSum, s, false,false,false).min)
  with_logger(JevoLogger()) do
      state = State("", rng,[comp_pop_creator, env_creator],
                    [InitializeAllPopulations(), 
                    AllVsAllMatchMaker(["p1", "p2"]),
                    Performer(time=true),
                    ComputeOutcomeMatrix(),
                    ElitistLexicaseSelectorAndReproducer(n_inds, ϵ=true, time=true),
                    min_sum_computer,
                    Reporter(GenotypeSum),
                    Mutator(),
                    ClearInteractionsAndRecords(), ], counters=counters)
      run!(state, 5)
  end
end

@testset "repeat sequence tfr lexicase" begin
    n_inds = 3
    n_tokens = 5
    seq_len = 3
    n_repeat = 4
    startsym = "<s>"
    endsym = "</s>"
    unksym = "<unk>"
    labels = string.(1:n_tokens)
    vocab = [unksym, startsym, endsym, labels...]
    vocab_size = length(vocab)

    n_blocks, n_heads, head_dim, hidden_dim, ff_dim = 1, 1, 1, 1, 1
    textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)

    attn_args = (n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim)
    block_args = (attn_args..., ff_dim=ff_dim)
    tfr_args = (block_args..., n_blocks=n_blocks, vocab_size=vocab_size)
    env_args = (n_labels = length(labels), batch_size = n_tokens ^ seq_len, seq_len = seq_len, n_repeat = n_repeat,)

    counters = default_counters()
    gene_counter = find(:type, AbstractGene, counters)
    tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
    developer = Creator(TransformerPhenotype, (;textenc=textenc))
    pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
    env_creator = Creator(RepeatSequence, env_args)

    Visualizer() = create_op("Reporter",
        retriever=Jevo.PopulationRetriever(),
        operator=(s,ps)-> ( ind = ps[1][1].individuals[1];
                           @info(string(ind.id)* " "*visualize(ind, ps[1][1]))))

    PrintOutcomeMatrix() = create_op("Reporter",
        retriever=Jevo.PopulationRetriever(),
        operator=(s,ps)-> (@info getonly(x->x isa Jevo.OutcomeMatrix, ps[1][1].data);))

    mrs = (0.1f0, 0.01f0, 0.001f0)
    state = State("", rng, [pop_creator, env_creator], 
                  [InitializeAllPopulations(),
                    CreateMissingWorkers(1, slurm=false),
                    InitializePhylogeny(),
                    InitializeDeltaCache(),
                    UpdateParentsAcrossAllWorkers(time=true),
                    SoloMatchMaker(["p"]), 
                    Performer(time=true),
                    ComputeOutcomeMatrix(),
                    PrintOutcomeMatrix(),
                    ElitistLexicaseSelectorAndReproducer(n_inds, ϵ=true),
                    Visualizer(),
                    Mutator(mr=mrs),
                    UpdatePhylogeny(),
                    UpdateDeltaCache(),
                    ClearInteractionsAndRecords(),
                ], counters=counters)
    run!(state, 2)
end
