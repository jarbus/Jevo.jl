k, n_inds = 1, 100
n_tokens, seq_len, n_repeat = 3, 3, 4
startsym, endsym, unksym = "<s>", "</s>", "<unk>"
labels = string.(0:n_tokens-1)
vocab = [unksym, startsym, endsym, labels...]
vocab_size = length(vocab)
n_blocks, n_heads, head_dim, hidden_dim, ff_dim = 1, 1, 4, 5, 4 # start out with 1 head
textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)
attn_args = (n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim)
block_args = (attn_args..., ff_dim=ff_dim)
tfr_args = (block_args..., n_blocks=n_blocks, vocab_size=vocab_size)
env_args = (n_labels = length(labels), batch_size = n_tokens^seq_len, seq_len = seq_len, n_repeat = n_repeat,)

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

@testset "units" begin
    #= counters = default_counters() =#
    #= gene_counter = find(:type, AbstractGene, counters) =#
    #= tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),)) =#
    #= developer = Creator(TransformerPhenotype, (;textenc=textenc)) =#
    #= pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters)) =#
    #= env_creator = Creator(RepeatSequence, env_args) =#

    @testset "test-add-head-unit" begin
        #= pop, delta, state = pop_creator(), tfr_gc(), State() =#
        #= double_mut_genome = delta.change #+ delta2 =#
        #= worker_genome = deepcopy(delta.change) =#
        #= # TODO confirm that four more weights are added =#
        #= blank_delta = Jevo.copyarchitecture(delta) =#
        #= double_mut_genome = delta.change + delta =#
        #= @test !any(Jevo.is_fresh, Jevo.map_get(blank_delta, Jevo.Weights)) =#
        #= for i in 1:4 =#
        #=     println("i: ", i) =#
        #=     println("worker_genome: ") =#
        #=     println(visualize(worker_genome)) =#
        #=     blank_delta = Jevo.copyarchitecture(worker_genome) =#
        #=     new_head_delta = Delta(Jevo.add_attention_head(rng, state, pop, blank_delta; prob=1.0, inits=(Jevo.apply_kaiming_normal_noise!,), tfr_args...)) =#
        #=     println("new_head_delta: ") =#
        #=     println(visualize(new_head_delta.change)) =#
        #=     worker_genome =Jevo.add_delta_to_genome(worker_genome, new_head_delta) =#
        #=     qkvw = worker_genome.layers[1].blocks[1].attention.layer.qkv.weights =#
        #=     p1 = Jevo.tensor(qkvw, weight_cache=Jevo.get_weight_cache()) =#
        #=     p2 = Jevo.tensor(qkvw, weight_cache=Jevo.get_weight_cache()) =#
        #=     @test hash(p1) == hash(p2) =#
        #= end =#
        # {q,k,v}_{weight,bias} + out_weight
        #= @test  sum(Jevo.is_fresh.(Jevo.map_get(new_delta, Jevo.Weights))) == 7 =#
        #= @test !all(Jevo.is_fresh.(Jevo.map_get(new_delta, Jevo.Weights))) =#

        #= for fn in (+, Jevo.add_delta_to_genome) =#
        #=     new_full_genome = fn(delta.change, new_delta) =#
        #=     @test new_full_genome.layers[1].embed.weights == new_full_genome.layers[1].embed.weights =#
        #=     @test  sum(Jevo.is_fresh.(Jevo.map_get(new_delta, Jevo.Weights))) == 7 =#
        #=     @test  any(Jevo.is_fresh, Jevo.map_get(new_full_genome, Jevo.Weights)) =#
        #=     @test !all(Jevo.is_fresh, Jevo.map_get(new_full_genome, Jevo.Weights)) =#
        #=     @test length(new_full_genome.layers[1].blocks) == 1 =#
        #= end =#
    end

    #= @testset "test-add-decoder-block-unit" begin =#
    #=     pop, delta, state = pop_creator(), tfr_gc(), State() =#
    #=     new_delta = Jevo.copyarchitecture(delta) =#
    #=     new_delta = Jevo.add_decoder_block(rng, state, pop, new_delta; prob=1.0, head_dims=(head_dim,), tfr_args...) =#
    #=     @test length(new_delta.change.layers[1].blocks) == n_blocks + 1 =#
    #=     @test sum(Jevo.is_fresh.(new_delta.change.layers[1].blocks)) == 1  =#
    #=     @test !all(Jevo.is_fresh.(Jevo.map_get(new_delta, Jevo.Weights))) =#
    #=     for fn in (+, Jevo.add_delta_to_genome) =#
    #=         new_full_genome = fn(delta.change, new_delta) =#
    #=         @test  any(Jevo.is_fresh, Jevo.map_get(new_full_genome, Jevo.Weights)) =#
    #=         @test !all(Jevo.is_fresh, Jevo.map_get(new_full_genome, Jevo.Weights)) =#
    #=         @test length(new_full_genome.layers[1].blocks) == n_blocks + 1 =#
    #=     end =#
    #= end =#
end


@testset "test-attention-head-add integration" begin
    counters = default_counters()
    gene_counter = find(:type, AbstractGene, counters)
    tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
    developer = Creator(TransformerPhenotype, (;textenc=textenc))
    pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
    env_creator = Creator(RepeatSequence, env_args)

    Visualizer(;kwargs...) = create_op("Reporter",
        retriever=Jevo.PopulationRetriever(),
        operator=(s,ps)-> ( ind = ps[1][1].individuals[1];
                           @info(string(ind.id)* " "*visualize(ind, ps[1][1]))); kwargs...)

    BestLogger(;kwargs...) = create_op("Reporter",
            retriever=Jevo.get_individuals,
            operator=(s,is)->
            (println("best id: $(is[1].id)");
                m=Measurement(NegativeLoss, evaluate(env_creator, is[1]), generation(s));
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
    run!(state, 40)
end
