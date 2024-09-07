@testset "incremental" begin
    n_inds = 3
    n_tokens = 5
    seq_len = 3
    n_repeat = 1
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
    env_args = (n_labels = length(labels), batch_size = n_tokens^seq_len, seq_len = seq_len, n_repeat = n_repeat,)

    counters = default_counters()
    gene_counter = find(:type, AbstractGene, counters)
    tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
    developer = Creator(TransformerPhenotype, (;textenc=textenc))
    pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
    env_creator = Creator(RepeatSequence, env_args)
    pop = pop_creator()
    @test !isdefined(Main, :best_scores)
    # play first ind
    interactions1 = play(env_creator, pop.individuals[2:2])
    @test isdefined(Main, :best_scores)
    intsum1 = sum(int.score for int in interactions1)
    @test sum(Main.best_scores) == intsum1
    @test intsum1 > -Inf
    # Play second individual, it performs worse than first ind, so terminate early 
    interactions2 = play(env_creator, pop.individuals[1:1])
    intsum2 = sum(int.score for int in interactions2)
    @test sum(Main.best_scores) == intsum1
    @test intsum2 == -Inf
end
