k, n_inds = 1, 100
n_tokens, seq_len, n_repeat = 3, 3, 4
startsym, endsym, unksym = "<s>", "</s>", "<unk>"
labels = string.(0:n_tokens-1)
vocab = [unksym, startsym, endsym, labels...]
vocab_size = length(vocab)
textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)
rnn_args = (dims=(32, 16), σ=relu)
env_args = (n_labels = length(labels), batch_size = n_tokens^seq_len, seq_len = seq_len, n_repeat = n_repeat,)

    


@testset "units" begin

    @testset "develop and forward pass" begin
        counters = default_counters()
        gene_counter = find(:type, AbstractGene, counters)
        rnn_geno = Jevo.RNN(rng, gene_counter, dims=(4, 5), σ=relu)
        get_weights(rnn_geno)
        rnn = Jevo.create_layer(rnn_geno, weight_cache=weight_cache)
        rnn(randn(Float32, 4))
    end

    #= counters = default_counters() =#
    #= gene_counter = find(:type, AbstractGene, counters) =#
    #= rnn_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.RNN, rnn_args)])),)) =#
    #= developer = Creator(Model, (;textenc=textenc)) =#
    #= pop_creator = Creator(Population, ("p", n_inds, PassThrough(rnn_gc), PassThrough(developer), counters)) =#
    #= env_creator = Creator(RepeatSequence, env_args) =#
    #==#
    #= Visualizer(;kwargs...) = create_op("Reporter", =#
    #=     retriever=Jevo.PopulationRetriever(), =#
    #=     operator=(s,ps)-> ( ind = ps[1][1].individuals[1]; =#
    #=                        @info(string(ind.id)* " "*visualize(ind, ps[1][1]))); kwargs...) =#
    #==#
    #= mrs = (0.1f0, 0.01f0, 0.001f0) =#
    #= state = State("", rng, [pop_creator, env_creator],  =#
    #=               [InitializeAllPopulations(), =#
    #=                 InitializePhylogeny(), =#
    #=                 InitializeDeltaCache(), =#
    #=                 SoloMatchMaker(["p"]),  =#
    #=                 Performer(time=true), =#
    #=                 ScalarFitnessEvaluator(), =#
    #=                 TruncationSelector(k), =#
    #=                 CloneUniformReproducer(n_inds), =#
    #=                 UpdatePhylogeny(), =#
    #=                 UpdateParentsAcrossAllWorkers(time=true), =#
    #=                 Visualizer(), =#
    #=                 ClearCurrentGenWeights(), =#
    #=                 NBackMutator(n_back=10, mrs=mrs, no_layer_norm=true, max_n_muts=3), =#
    #=                 UpdateDeltaCache(), =#
    #=                 ClearInteractionsAndRecords(), =#
    #=             ], counters=counters) =#
    #= run!(state, 5) =#
end
