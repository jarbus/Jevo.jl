state = State()
rng = StableRNG(1)
gene_counter = Jevo.get_counter(AbstractGene, state)

nul_pop = Population("", Individual[])
n_tokens = 5
seq_len = 3
n_repeat = 4
startsym = "<s>"
endsym = "</s>"
unksym = "<unk>"
labels = string.(1:n_tokens)
vocab = [unksym, startsym, endsym, labels...]
vocab_size = length(vocab)

n_blocks, n_heads, head_dim, hidden_dim, ff_dim = 3, 2, 5, 10, 20
textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)

attn_args = (n_heads=n_heads, head_dim=head_dim, hidden_dim=hidden_dim)
block_args = (attn_args..., ff_dim=ff_dim)
tfr_args = (block_args..., n_blocks=n_blocks, vocab_size=vocab_size)
env_args = (n_labels = length(labels), batch_size = n_tokens^seq_len, seq_len = seq_len, n_repeat = n_repeat,)


k = 1
n_inds = 2
developer = Creator(TransformerPhenotype, (;textenc=textenc))
tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
counters = default_counters()
parent_geno = tfr_gc()
parent_weights = Jevo.get_weights(parent_geno, no_layer_norm=true)
n_mutation_targets = length(Jevo.get_weights(parent_geno, no_layer_norm=true))


#@testset "Transformer Delta+Reproduce+Mutate" begin
#
#@testset "Integrate mutation + reproduction" begin
#    # Integration test
#    pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
#    state = State("", rng, [pop_creator], 
#                  [InitializeAllPopulations(),
#                   InitializePhylogeny(),
#                   InitializeDeltaCache(),
#                   RandomEvaluator(),
#                   TruncationSelector(k),
#                   CloneUniformReproducer(n_inds),
#                   Mutator(;mr=Float32(0.01), n=-1),
#                   UpdatePhylogeny(),
#                   UpdateDeltaCache(),
#                   ClearInteractionsAndRecords(),
#                  ], counters=counters)
#    run!(state, 1)
#    parents = state.populations[1].individuals |> x->filter(i->i.generation==0, x)
#    children = state.populations[1].individuals |> x->filter(i->i.generation==1, x)
#    @test length(parents) == k
#    @test length(children) == n_inds - k
#    # test that each parent has a mutation of MR=1 and each child has a mutation of MR=0.01
#    dc = Jevo.get_delta_cache(state.populations[1])
#    for parent in parents, w in Jevo.get_weights(parent.genotype, no_layer_norm=true)
#        @test length(w.muts) == 1
#        @test w.muts[1].mr == 1
#        @test parent.id ∈ keys(dc)
#    end
#    for child in children, w in Jevo.get_weights(child.genotype, no_layer_norm=true)
#        @test length(w.muts) == 1
#        @test w.muts[1].mr == 0.01f0
#        @test child.id ∈ keys(dc)
#    end
#end
#end
#
using Distributed
addprocs(1)
@testset "Distribute and Develop" begin
    @everywhere begin
        using Jevo, Flux
        weight_cache = WeightCache(maxsize=Int(1e7))
        genotype_cache = GenotypeCache(maxsize=Int(1e7))
        Jevo.set_device()
    end
    @test myid() == 1
    """
    two steps:
    1. ensure all nodes have parents
      a. master sends (grandparent id, parent id, parent delta) pairs
      b. worker requests missing parent genotypes they need AND constructs parents
      c. master responds with full parent genotypes AND outputs warning
      d. worker constructs and caches parents in genotype cache and weight matrix
    2. send deltas & construct
      a. master sends delta
      b. worker constructs genotype
    
    setting:
    
    a
    |
    b
    |
    c
    """
    
    # setup
    state = State()
    rng = state.rng
    gene_counter = Jevo.get_counter(AbstractGene, state)

    vocab = ["1", "2", "3"]
    startsym, endsym, unksym = "1", "1", "1"
    textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)
    developer = Creator(TransformerPhenotype, (;textenc=textenc))
    tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))

    # genesis
    a = Individual(state.counters, tfr_gc, developer)
    p = Population("p", [a])
    push!(state.populations, p)
    Jevo.operate!(state, InitializePhylogeny())
    Jevo.operate!(state, InitializeDeltaCache())
    Jevo.operate!(state, UpdateDeltaCache())
    tree, dc = Jevo.get_tree(p), Jevo.get_delta_cache(p)
    # Unit test
    @test (-1, -1, nothing) == Jevo.master_get_gpid_pid_pds(a, tree, dc)
    # Confirm p2 doesn't have `a` in it's genotype cache
    @test fetch(@spawnat 2 !haskey(Jevo.get_genotype_cache(), a.id))
    workers_missing_parents = Jevo.master_send_pids_and_gpids([[p]])
    @test workers_missing_parents[2] == Int[]
    # Confirm p2 still doesn't have `a` in it's genotype cache
    @test fetch(@spawnat 2 !haskey(Jevo.get_genotype_cache(), a.id))
    # test constructing a, genesis node
    @test a.genotype.change == Jevo.master_construct_genome(a, Jevo.get_tree(p),
                                         Jevo.get_delta_cache(p),
                                         Jevo.get_genotype_cache())

    # Construct full parent genome (only one node)
    b = Jevo.clone(state, a)
    @test b.parents[1] == a.id
    b.genotype = mutate(rng, state, nul_pop, b.genotype, mr=0.1f0)
    push!(p.individuals, b)
    Jevo.operate!(state, UpdatePhylogeny())
    @test tree.tree[b.id].parent.id == a.id
    @test tree.tree[a.id].children[1].id == b.id
    Jevo.operate!(state, UpdateDeltaCache())
    Jevo.operate!(state, Jevo.GenerationIncrementer())
    @test (-1, a.id, a.genotype) == Jevo.master_get_gpid_pid_pds(b, tree, dc)
    # Confirm p2 still doesn't have `a` or `b` in it's genotype cache
    @test fetch(@spawnat 2 !haskey(Jevo.get_genotype_cache(), a.id))
    @test fetch(@spawnat 2 !haskey(Jevo.get_genotype_cache(), b.id))
    workers_missing_parents = Jevo.master_send_pids_and_gpids([[p]])
    @test workers_missing_parents[2] == [a.id]
    # confirm p2 doesn't have `a` or `b`
    @test fetch(@spawnat 2 !haskey(Jevo.get_genotype_cache(), a.id))
    @test fetch(@spawnat 2 !haskey(Jevo.get_genotype_cache(), b.id))

    # construct missing parent genomes
    worker_parent_genomes = Jevo.master_construct_parents_genomes([[p]], workers_missing_parents)
    @test worker_parent_genomes[2] == [(a.id, a.genotype.change)]

    # send missing parent genomes to worker
    Jevo.master_cache_parents!(worker_parent_genomes) 
    # confirm p2 has `a` but not `b`
    @test fetch(@spawnat 2 haskey(Jevo.get_genotype_cache(), a.id))
    @test fetch(@spawnat 2 !haskey(Jevo.get_genotype_cache(), b.id))
    # test genotypes are the same
    @test fetch(@spawnat 2 Jevo.get_genotype_cache()[a.id]) == a.genotype.change

    # now add and construct c
    c = Jevo.clone(state, b)
    c.genotype = mutate(rng, state, nul_pop, c.genotype, mr=0.1f0)
    push!(p.individuals, c)
    Jevo.operate!(state, UpdatePhylogeny())
    Jevo.operate!(state, UpdateDeltaCache())
    Jevo.operate!(state, Jevo.GenerationIncrementer())

    @test (a.id, b.id, b.genotype) == Jevo.master_get_gpid_pid_pds(c, tree, dc)
    workers_missing_parents = Jevo.master_send_pids_and_gpids([[p]])
    # we reconstruct a because it's parent still isn't on the worker, only a is.
    # This is technically wasteful, but far less code and not that big of a deal
    @test workers_missing_parents[2] == Int[a.id]
    # Confirm `b` is on the worker
    @test fetch(@spawnat 2 !haskey(Jevo.get_genotype_cache(), c.id))
    @test fetch(@spawnat 2 haskey(Jevo.get_genotype_cache(), b.id))

    # construct c on worker
    c_genotype = fetch(@spawnat 2 Jevo.worker_construct_child_genome(c))
    # confirm that there are the same number of mutations in a + b + c
    # as there are in reconstructed c
    ind_weights = []
    for ind in (a, b, c)
        push!(ind_weights, [])
        for w in Jevo.get_weights(ind.genotype)
            push!(ind_weights[end], length(w.muts))
        end
    end
    c_weights = []
    for w in Jevo.get_weights(c_genotype)
        push!(c_weights, length(w.muts))
    end
    for idx in eachindex(c_weights) 
        n_muts = 0
        for ind_weight in ind_weights
            n_muts += ind_weight[idx]
        end
        @test n_muts == c_weights[idx]
    end
    addprocs(1)
    @test workers() == [2,3]
    @everywhere begin
        using Jevo, Flux
        weight_cache = WeightCache(maxsize=Int(1e7))
        genotype_cache = GenotypeCache(maxsize=Int(1e7))
        # don't set jevo device here since we might only have one gpu
        Main.jevo_device = Flux.get_device("CUDA", 0)
    end
    Jevo.operate!(state, UpdateParentsAcrossAllWorkers())
    c_genotype_w3 = fetch(@spawnat 3 Jevo.worker_construct_child_genome(c))
    @test c_genotype == c_genotype_w3
    # test development of all genos on worker
    # TODO fill this out once I figure out a cleaner way to test on gpu
    # for ind in (a, b, c)
    #     @test ind isa Individual
    #     pheno_2 = fetch(@spawnat 2 develop(developer, ind))
    #     pheno_3 = fetch(@spawnat 3 develop(developer, ind))
    #     @test Flux.params(pheno_2.trf) == Flux.params(pheno_3.trf) 
    # end
end
#
#abstract type OutputValue <: AbstractMetric end
#abstract type MatchCount <: AbstractMetric end
#
#
#@testset "Insane integration" begin
#    k = 1
#    n_inds = 10
#
#    @everywhere begin
#        Main.weight_cache = WeightCache(maxsize=Int(1e10))
#        Main.genotype_cache = GenotypeCache(maxsize=Int(1e10))
#        Main.jevo_device = Flux.get_device("CUDA", 0)
#    end
#
#    counters = default_counters()
#    rng = StableRNG(1) 
#    gene_counter = find(:type, AbstractGene, counters)
#    developer = Creator(TransformerPhenotype, (;textenc=textenc))
#    tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
#
#    function evaluate(individual)
#      # geno_hash = hash(get_weights(individual.genotype, no_layer_norm=true))
#      # println("geno_hash: $geno_hash")
#      model = develop(individual.developer, individual)
#      # pheno_hash = Flux.params(model.trf) |> Iterators.flatten |> collect |> hash
#      # println("pheno_hash: $pheno_hash")
#      Jevo.play(env_creator(), [model])[1]
#    end
#
#    losses = []
#    BestLogger() = create_op("Reporter",
#            retriever=Jevo.get_individuals,
#            operator=(s,is)->
#            (println("best id: $(is[1].id)");
#               m=StatisticalMeasurement(OutputValue, evaluate.(is), generation(s));
#               push!(losses, m.mean);
#               println(Jevo.get_weight_symbols(is[1]));
#              @info m;))
#
#    pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
#    env_creator = Creator(RepeatSequence, env_args)
#
#    state = State("", rng, [pop_creator, env_creator], 
#                  [InitializeAllPopulations(),
#                   InitializePhylogeny(),
#                   InitializeDeltaCache(),
#                   UpdateParentsAcrossAllWorkers(),
#                   SoloMatchMaker(["p"]), 
#                   Performer(time=true),
#                   ScalarFitnessEvaluator(["p"]), 
#                   TruncationSelector(k),
#                   BestLogger(),
#                   CloneUniformReproducer(n_inds),
#                   Mutator(mr=0.1f0),
#                   UpdatePhylogeny(),
#                   UpdateDeltaCache(),
#                   ClearInteractionsAndRecords(),
#                  ], counters=counters)
#
#    run!(state, 10)
#    println(losses)
#    addprocs(1)
#    @everywhere begin
#        using Jevo, Flux
#        weight_cache = WeightCache(maxsize=Int(1e10))
#        genotype_cache = GenotypeCache(maxsize=Int(1e10))
#        Main.jevo_device = Flux.get_device("CUDA", 0)
#    end
#    run!(state, 2)
#end

@testset "insaner integration tests" begin
    n_inds = 3
    genepool_start = 4
    n_latest = 1
    counters = default_counters()
    gene_counter = find(:type, AbstractGene, counters)
    tfr_gc = Creator(Delta, (Creator(Network, (rng, gene_counter, [(Jevo.Transformer, tfr_args)])),))
    developer = Creator(TransformerPhenotype, (;textenc=textenc))
    pop_creator = Creator(Population, ("p", n_inds, PassThrough(tfr_gc), PassThrough(developer), counters))
    env_creator = Creator(RepeatSequence, env_args)
    
    function verify(state::State, pop::Population)
        
        gen = generation(state)
        gp = Jevo.getonly(d->d isa Jevo.GenePool, pop.data)
        
        tree = Jevo.get_tree(pop)
        cache = Jevo.get_delta_cache(pop)
        @test cache isa DeltaCache
        @test length(cache) == length(tree.tree)
        
        for ind in pop.individuals # test that each individual has a delta
            @test any(w->!isempty(w.muts), Jevo.get_weights(ind.genotype, no_layer_norm=true))
        end
        
        if gen > 1
            @test length(gp.deltas) > 0
            latest_genotypes = Set(ind.genotype for ind in pop.individuals if ind.generation == gen)
            for d in gp.deltas
                @test d ∉ latest_genotypes
            end
        end
    end

    abstract type OutputValue <: AbstractMetric end

    function evaluate(individual)
      percent_correct = fetch(@spawnat(2, begin

        function percentage_evaluation_npeat(trf::TransformerPhenotype; n::Int, kwargs...)
            n_perfect = 0
            for i in 0:n-1, j in 0:n-1, k in 0:n-1
                prompt = "$i $j $k $i $j $k"
                full_str = infer(trf, prompt; kwargs...)[1]
                if length(full_str) >= 27
                  n_perfect += full_str[5:15] == full_str[17:27]
                  @info full_str
                else
                end
            end
            n_perfect / n^3
        end
        model = develop(developer, individual)
        percentage_evaluation_npeat(model, n=n_tokens, max_len=15)
      end))
      @info "Percentage perfect: $(round(percent_correct, digits=3))"
      fetch(@spawnat(2, begin
          device!(Main.jevo_device_id)
          mean(interaction.score for interaction in Jevo.play(env_creator, [individual]))
        end))
    end

    meu(interactions) = sum(int.score for int in interactions)


    function meu_individual(inds)
        @assert length(inds) > 0
        meu_max = -Inf
        meu_ind = nothing
        for ind in inds
            length(ind.interactions) == 0 && continue  # skip new inds without interactions
            new_meu = meu(ind.interactions)
            new_meu <= meu_max && continue
            meu_ind = ind
            meu_max = new_meu
        end
        @assert !isnothing(meu_ind)
        @info "found meu_ind $(meu_ind.id)"
        meu_ind
    end

    BestReporter() = create_op("Reporter",
            retriever=Jevo.get_individuals,
            operator=(s,is)->
            (m= StatisticalMeasurement(OutputValue, [evaluate(meu_individual(is))], generation(s));
              @info m; @h5 m;),
            time=true)
    BestVisualizer() = create_op("Reporter",
            retriever=Jevo.PopulationRetriever(),
            operator=(s,ps)-> (meu_ind = meu_individual(ps[1][1].individuals);
                               @info(string(meu_ind.id)* " "*visualize(meu_ind, ps[1][1]))))
    verifier = create_op("Assertor",
            condition=s->generation(s) > genepool_start,
            retriever=PopulationRetriever(),
            operator=map(map((s,p)->verify(s,p))))
        
    mrs = (0.1f0, 0.01f0, 0.001f0)
    state = State("", rng, [pop_creator, env_creator], 
                  [InitializeAllPopulations(),
                    CreateMissingWorkers(1, slurm=false),
                    InitializePhylogeny(),
                    InitializeDeltaCache(),
                    UpdateParentsAcrossAllWorkers(time=true),
                    SoloMatchMaker(["p"]), 
                    Performer(time=true),
                    ScalarFitnessEvaluator(["p"]), 
                    TruncationSelector(k),
                    BestReporter(),
                    ClearCurrentGenWeights(),
                    UpdateGenePool(n_latest=n_latest, after_gen=genepool_start),
                    ComputeGenePoolNetwork(after_gen=genepool_start, no_layer_norm=false, time=true),
                    BestVisualizer(),
                    CloneUniformReproducer(n_inds),
                    Mutator(mr=mrs, condition=s->generation(s) <= genepool_start),
                    NNGenePoolMutator(mr=mrs, after_gen=genepool_start, no_layer_norm=true),
                    UpdatePhylogeny(),
                    UpdateDeltaCache(),
                    verifier,
                    ClearInteractionsAndRecords(),
                ], counters=counters)
    run!(state, 2)
end
