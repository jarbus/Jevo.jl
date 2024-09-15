k = 1
n_inds = 10
startsym, endsym, unksym = "<s>", "</s>", "<unk>"
mrs = (0.1f0, 0.01f0, 0.001f0)

function get_fsa_components(seq_idx::Int)
    seq_len = 8
    n_strings = 8
    labels = ["0", "1", ":", "r", "a"]
    vocab = [unksym, startsym, endsym, labels...]
    vocab_size = length(vocab)
    n_blocks, n_heads, head_dim, hidden_dim, ff_dim = 1, 4, 4, 4, 4
    env_type = seq_idx ∈ (1, 2, 7) ? RegularLanguage : AcceptRejectStrings
    env_configs = Dict(
        1 => (regex=r"^1*$", seq_len=seq_len, n_strings=n_strings),
        2 => (regex=r"^(10)+$", seq_len=seq_len, n_strings=n_strings),
        7 => (regex=r"^0*1*0*1*$", seq_len=seq_len, n_strings=n_strings),
        3 => (accept=["1",  "0",  "01",  "11",  "00",  "100",  "1", "10",  "1", "11",  "0", "00",  "1", "0", "0", "10", "0",  "110000011100001",  "111101100010011100",], reject=["10",  "101",  "0", "10",  "1", "0", "10",  "1", "110",  "1", "0", "11",  "10", "0", "01",  "1", "110", "10",  "1", "0", "0", "10", "0", "0",  "1", "11110", "0", "0",  "0", "1110", "0", "110", "1",  "1", "10", "1110", "0", "110"]),
        5 => ( accept=["11",  "00",  "10", "01",  "0", "101",  "1", "0", "10",  "1", "0", "0", "0", "111101",  "1001100001111010",  "1", "111", "11",  "0", "0", "00",], reject=["0",  "1", "11",  "0", "10",  "000000000",  "1000",  "01",  "10",  "1", "110", "0", "10", "10", "0",  "010111111110",  "0", "0", "01",  "0", "11"])
    )
    env_args = env_configs[seq_idx]
    env_creator = Creator(env_type, env_args)
    textenc = TransformerTextEncoder(x->split(x,""), vocab; startsym, endsym, unksym, padsym=unksym)
    block_args = (n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim, ff_dim=ff_dim)
    tfr_args = (block_args..., n_blocks=n_blocks, vocab_size=vocab_size)
    counters = default_counters()
    gene_counter = find(:type, AbstractGene, counters)
    tfr_gc = Creator(Delta, Creator(TextTransformer, (rng, gene_counter, tfr_args)))
    developer = Creator(TextModel, (;textenc=textenc))
    pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
    pop_creator, env_creator, tfr_args, counters
end

@testset "test-fsa" begin
    for seq_idx in (1,2,3,5,7)
        @everywhere begin
            Main.weight_cache = nothing
            Main.genotype_cache = nothing
        end
        pop_creator, env_creator, tfr_args, counters = get_fsa_components(seq_idx)
        state = State("", rng, [pop_creator, env_creator], 
                      [InitializeAllPopulations(),
                        CreateMissingWorkers(1, slurm=false, c=2),
                        InitializePhylogeny(),
                        InitializeDeltaCache(),
                        SoloMatchMaker(["p"]), 
                        Performer(time=true),
                        ScalarFitnessEvaluator(),
                        TruncationSelector(k),
                        CloneUniformReproducer(n_inds),
                        UpdatePhylogeny(),
                        UpdateParentsAcrossAllWorkers(time=true),
                        #= Visualizer(condition=s->generation(s) % 10 == 0), =#
                        RecordPerformance(env_creator, condition=s->generation(s) % 1 == 0), 
                        ClearCurrentGenWeights(),
                        NBackMutator(n_back=3, mrs=mrs, max_n_muts=2, no_layer_norm=true),
                        #= AddAttentionHeads(prob=0.05, inits=(Jevo.apply_kaiming_normal_noise!,)), =#
                        #= AddDecoderBlock(;prob=0.05, head_dims=(tfr_args.head_dim,), tfr_args...), =#
                        UpdateDeltaCache(),
                        ClearInteractionsAndRecords(),
                    ], counters=counters)
        rmprocs(workers())
        run!(state, 5)
    end
end
