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

@testset "units" begin

    counters = default_counters()
    gene_counter = find(:type, AbstractGene, counters)

    rnn_geno = Jevo.RNN(rng, gene_counter, dims=(4, 5), σ=relu)
    println(length(get_weights(rnn_geno)))
    rnn = Jevo.create_layer(rnn_geno, weight_cache)
    rnn(randn(Float32, 4))

    #= tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),)) =#
    #= developer = Creator(TransformerPhenotype, (;textenc=textenc)) =#
    #= pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters)) =#
    #= env_creator = Creator(RepeatSequence, env_args) =#




    #= @testset "nbackmutator" begin =#
    #==#
    #=     counters = default_counters() =#
    #=     gene_counter = find(:type, AbstractGene, counters) =#
    #=     tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),)) =#
    #=     developer = Creator(TransformerPhenotype, (;textenc=textenc)) =#
    #=     pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters)) =#
    #=     env_creator = Creator(RepeatSequence, env_args) =#
    #==#
    #=     Visualizer(;kwargs...) = create_op("Reporter", =#
    #=         retriever=Jevo.PopulationRetriever(), =#
    #=         operator=(s,ps)-> ( ind = ps[1][1].individuals[1]; =#
    #=                            @info(string(ind.id)* " "*visualize(ind, ps[1][1]))); kwargs...) =#
    #==#
    #=     mrs = (0.1f0, 0.01f0, 0.001f0) =#
    #=     state = State("", rng, [pop_creator, env_creator],  =#
    #=                   [InitializeAllPopulations(), =#
    #=                     InitializePhylogeny(), =#
    #=                     InitializeDeltaCache(), =#
    #=                     SoloMatchMaker(["p"]),  =#
    #=                     Performer(time=true), =#
    #=                     ScalarFitnessEvaluator(), =#
    #=                     TruncationSelector(k), =#
    #=                     CloneUniformReproducer(n_inds), =#
    #=                     UpdatePhylogeny(), =#
    #=                     UpdateParentsAcrossAllWorkers(time=true), =#
    #=                     Visualizer(), =#
    #=                     ClearCurrentGenWeights(), =#
    #=                     NBackMutator(n_back=10, mrs=mrs, no_layer_norm=true, max_n_muts=3), =#
    #=                     UpdateDeltaCache(), =#
    #=                     ClearInteractionsAndRecords(), =#
    #=                 ], counters=counters) =#
    #=     run!(state, 10) =#
    #= end =#
end

#= @testset "test-attention-head-add integration" begin =#
#=     counters = default_counters() =#
#=     gene_counter = find(:type, AbstractGene, counters) =#
#=     tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),)) =#
#=     developer = Creator(TransformerPhenotype, (;textenc=textenc)) =#
#=     pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters)) =#
#=     env_creator = Creator(RepeatSequence, env_args) =#
#==#
#=     Visualizer(;kwargs...) = create_op("Reporter", =#
#=         retriever=Jevo.PopulationRetriever(), =#
#=         operator=(s,ps)-> ( ind = ps[1][1].individuals[1]; =#
#=                            @info(string(ind.id)* " "*visualize(ind, ps[1][1]))); kwargs...) =#
#==#
#=     BestLogger(;kwargs...) = create_op("Reporter", =#
#=             retriever=Jevo.get_individuals, =#
#=             operator=(s,is)-> =#
#=             (println("best id: $(is[1].id)"); =#
#=                 m=Measurement(NegativeLoss, evaluate(env_creator, is[1]), generation(s)); =#
#=               @info m;); kwargs...) =#
#==#
#==#
#=     mrs = (0.1f0, 0.01f0, 0.001f0) =#
#=     state = State("", rng, [pop_creator, env_creator],  =#
#=                   [InitializeAllPopulations(), =#
#=                     CreateMissingWorkers(1, slurm=false), =#
#=                     InitializePhylogeny(), =#
#=                     InitializeDeltaCache(), =#
#=                     SoloMatchMaker(["p"]),  =#
#=                     Performer(time=true), =#
#=                     ScalarFitnessEvaluator(), =#
#=                     TruncationSelector(k), =#
#=                     CloneUniformReproducer(n_inds), =#
#=                     UpdatePhylogeny(), =#
#=                     UpdateParentsAcrossAllWorkers(time=true), =#
#=                     Visualizer(condition=s->generation(s) % 10 == 0), =#
#=                     BestLogger(condition=s->generation(s) % 1 == 0), =#
#=                     ClearCurrentGenWeights(), =#
#=                     NBackMutator(n_back=40, mrs=mrs, no_layer_norm=true), =#
#=                     AddAttentionHeads(prob=0.05, inits=(Jevo.apply_kaiming_normal_noise!,)), =#
#=                     AddDecoderBlock(;prob=0.05, head_dims=(head_dim,), tfr_args...), =#
#=                     UpdateDeltaCache(), =#
#=                     ClearInteractionsAndRecords(), =#
#=                 ], counters=counters) =#
#=     run!(state, 40) =#
#= end =#
