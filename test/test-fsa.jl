@testset "test-fsa" begin
    k = 1
    n_inds = 100
    seq_len = 3
    n_strings =6
    startsym = "<s>"
    endsym = "</s>"
    unksym = "<unk>"
    labels = ["0", "1", ":", "r", "a"]
    vocab = [unksym, startsym, endsym, labels...]
    vocab_size = length(vocab)
    n_blocks, n_heads, head_dim, hidden_dim, ff_dim = 1, 4, 4, 4, 4
    #= env_args = (regex=r"^(0|1)*0$", seq_len=seq_len, n_strings=2) =#
    env_args = (regex=r"^1*$", seq_len=seq_len, n_strings=n_strings)

    textenc = TransformerTextEncoder(x->split(x,""), vocab; startsym, endsym, unksym, padsym=unksym)
    attn_args = (n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim)
    block_args = (attn_args..., ff_dim=ff_dim)
    tfr_args = (block_args..., n_blocks=n_blocks, vocab_size=vocab_size)

    counters = default_counters()
    gene_counter = find(:type, AbstractGene, counters)
    tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
    developer = Creator(TransformerPhenotype, (;textenc=textenc))
    pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
    env_creator = Creator(RegularLanguage, env_args)
    abstract type OutputValue <: AbstractMetric end

    Visualizer(;kwargs...) = create_op("Reporter",
        retriever=Jevo.PopulationRetriever(),
        operator=(s,ps)-> ( ind = ps[1][1].individuals[1];
                           @info(string(ind.id)* " "*visualize(ind, ps[1][1]))), kwargs...)

    BestLogger(;kwargs...) = create_op("Reporter",
            retriever=Jevo.get_individuals,
            operator=(s,is)->
            (println("best id: $(is[1].id)");
                m=Measurement(OutputValue, evaluate(env_creator, is[1]), generation(s));
              @info m;); kwargs...)


    mrs = (0.1f0, 0.01f0, 0.001f0)
    state = State("", rng, [pop_creator, env_creator], 
                  [InitializeAllPopulations(),
                    CreateMissingWorkers(1, slurm=false),
                    InitializePhylogeny(),
                    InitializeDeltaCache(),
                    UpdateParentsAcrossAllWorkers(time=true),
                    SoloMatchMaker(["p"]), 
                    Performer(time=true),
                    ScalarFitnessEvaluator(),
                    TruncationSelector(k),
                    CloneUniformReproducer(n_inds),
                    ClearCurrentGenWeights(),
                    Visualizer(),
                    BestLogger(condition=s->generation(s) % 10 == 0),
                    Mutator(mr=mrs),
                    UpdatePhylogeny(),
                    UpdateDeltaCache(),
                    ClearInteractionsAndRecords(),
                ], counters=counters)
    run!(state, 100)
end
