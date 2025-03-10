function Jevo.SelfAttention(rng::Jevo.AbstractRNG, counter::Jevo.AbstractCounter; n_heads::Int, head_dim::Int, hidden_dim::Int, qkv_rank::Int=-1, o_rank::Int=-1, init!::Function=Jevo.apply_kaiming_normal_noise!,)
    # =============================================================================================
    # NOTE: QKV weights are transposed, because we aren't going through our custom Dense constructor
    #       which automatically transposes for us.
    # =============================================================================================
    head_init! = qkv_rank < 1 ? init! : Jevo.apply_kaiming_normal_noise_factored!
    head_weights = Jevo.WeightsCollection(
        (head_dim*n_heads*3, hidden_dim),
        vcat([Jevo.Weights(rng, counter, (head_dim, hidden_dim), init=head_init!) for i in 1:n_heads*3]))

    factors = Jevo.FactorWeight(
        (head_dim*n_heads*3, hidden_dim),
        Jevo.Weights(rng, counter, (head_dim*n_heads*3, qkv_rank), init=init!),
        Jevo.Weights(rng, counter, (qkv_rank, hidden_dim), init=init!)
    )
    
    qkv_weight = qkv_rank < 1 ? head_weights : Jevo.CompositeWeight((hidden_dim, head_dim*n_heads*3), [head_weights, factors])

    qkv_bias = Jevo.WeightsCollection(
        (head_dim*n_heads*3,),
        [Jevo.Weights(rng, counter, (head_dim,), init=init!) for i in 1:n_heads*3]
    )

    out_weight = Jevo.WeightsCollection(
        (hidden_dim, head_dim*n_heads),
        [Jevo.Weights(rng, counter, (hidden_dim, head_dim), init=head_init!) for _ in 1:1, h in 1:n_heads])

    out_bias = Jevo.Weights(rng, counter, (hidden_dim,), init=init!)
    
    qkv = Jevo.Dense(qkv_weight, qkv_bias, identity)
    out = Jevo.Dense(out_weight, out_bias, identity)

    Jevo.SelfAttention(n_heads, qkv, out)
end



@testset "test-fsa" begin
    k = 1
    n_inds = 200
    seq_len = 8
    n_strings = 8
    startsym = "<s>"
    endsym = "</s>"
    unksym = "<unk>"
    labels = ["0", "1", ":", "r", "a"]
    vocab = [unksym, startsym, endsym, labels...]
    vocab_size = length(vocab)
    n_blocks, n_heads, head_dim, hidden_dim, ff_dim = 1, 4, 4, 4, 4
    #= env_args = (regex=r"^1*$", seq_len=seq_len, n_strings=n_strings) # 1st sequence =#
    #= env_args = (regex=r"^(10)+$", seq_len=seq_len, n_strings=n_strings) # 2nd sequence =#
    #= env_args = (regex=r"^0*1*0*1*$", seq_len=seq_len, n_strings=n_strings) # 7th sequence =#
    #= env_creator = Creator(RegularLanguage, env_args) =#
    #= env_args = (accept=["1",  "0",  "01",  "11",  "00",  "100",  "1", "10",  "1", "11",  "0", "00",  "1", "0", "0", "10", "0",  "110000011100001",  "111101100010011100",], reject=["10",  "101",  "0", "10",  "1", "0", "10",  "1", "110",  "1", "0", "11",  "10", "0", "01",  "1", "110", "10",  "1", "0", "0", "10", "0", "0",  "1", "11110", "0", "0",  "0", "1110", "0", "110", "1",  "1", "10", "1110", "0", "110"]) #   3rd sequence =#
    env_args = ( accept=["11",  "00",  "10", "01",  "0", "101",  "1", "0", "10",  "1", "0", "0", "0", "111101",  "1001100001111010",  "1", "111", "11",  "0", "0", "00",], reject=["0",  "1", "11",  "0", "10",  "000000000",  "1000",  "01",  "10",  "1", "110", "0", "10", "10", "0",  "010111111110",  "0", "0", "01",  "0", "11",],)   # 5th sequence
    env_creator = Creator(AcceptRejectStrings, env_args)

    textenc = TransformerTextEncoder(x->split(x,""), vocab; startsym, endsym, unksym, padsym=unksym)
    attn_args = (n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim)
    block_args = (attn_args..., ff_dim=ff_dim)
    tfr_args = (block_args..., n_blocks=n_blocks, vocab_size=vocab_size)

    counters = default_counters()
    gene_counter = find(:type, AbstractGene, counters)
    tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
    developer = Creator(TransformerPhenotype, (;textenc=textenc))
    pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
    abstract type OutputValue <: AbstractMetric end

    Visualizer(;kwargs...) = create_op("Reporter",
        retriever=Jevo.PopulationRetriever(),
        operator=(s,ps)-> ( ind = ps[1][1].individuals[1];
                           @info(string(ind.id)* " "*visualize(ind, ps[1][1]))); kwargs...)

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
                    SoloMatchMaker(["p"]), 
                    Performer(time=true),
                    ScalarFitnessEvaluator(),
                    TruncationSelector(k),
                    CloneUniformReproducer(n_inds),
                    UpdatePhylogeny(),
                    UpdateParentsAcrossAllWorkers(time=true),
                    Visualizer(condition=s->generation(s) % 10 == 0),
                    BestLogger(condition=s->generation(s) % 1 == 0),
                    ClearCurrentGenWeights(),
                    NBackMutator(n_back=40, mrs=mrs, no_layer_norm=true),
                    AddAttentionHeads(prob=0.05, inits=(Jevo.apply_kaiming_normal_noise!,)),
                    AddDecoderBlock(;prob=0.05, head_dims=(head_dim,), tfr_args...),
                    UpdateDeltaCache(),
                    ClearInteractionsAndRecords(),
                ], counters=counters)
    run!(state, 80)
end
