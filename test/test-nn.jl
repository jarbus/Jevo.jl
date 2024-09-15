nul_pop = Population("", Individual[])
@testset "neural-net" begin
    @testset "unit tests" begin
        state = State()
        gene_counter = Jevo.get_counter(AbstractGene, state)
        rng = StableRNG(1)
        @test gene_counter |> value == 1
        # test creating a genotype
        relu(x) = max(0, x)
        # Naive way to create network
        Jevo.Chain([Jevo.Dense(Jevo.Weights((784,784),NetworkGene[]), Jevo.Weights((784,10),NetworkGene[]), relu)])
        # Better interface
        net = Jevo.Chain(rng, gene_counter, [(Jevo.Dense, (dims=(784,10), σ=relu))])
        dense = net.layers[1]

        @testset "tensor()" begin
            @test (10,784) == size(Jevo.tensor(dense.weights))
            @test mean(Jevo.tensor(dense.weights)) ≈ 0.0 atol=0.01
        end
        @testset "apply_sparse_noise!()" begin
            arr = zeros(Float32, 784, 10)
            Jevo.apply_sparse_noise!(rng, Float32, arr, 0.1f0)
            # confirm that the array is not all zeros
            @test !all(iszero, arr)
            @test any(iszero, arr)


        end

        @testset "WeightsCollection" begin

            @testset "Matrix Errors" begin
                dims = (784, 15)

                bad_breakdowns = [(784,10) (784, 4);], # dimension mismatch
                                 [(784,10) (7, 5);],   # row mismatch
                                 [(200,10) (100, 5);
                                  (584,10) (684, 5)]   # Either columns or rows must match
                for breakdown in bad_breakdowns
                    try
                        Jevo.WeightsCollection(rng, gene_counter, dims=dims, breakdown=breakdown)
                        @warn "Should have thrown an error for breakdown: $breakdown"
                        @test false 
                    catch
                        @test true
                    end
                end

                valid_breakdowns = [(784,10) (784, 5);], # rows match
                                   [(384,15); (400,15)],     # columns match
                                   [(400,10) (400, 5);
                                    (384,10) (384, 5)] # Either columns or rows must match
                
                for breakdown in valid_breakdowns
                    Jevo.WeightsCollection(rng, gene_counter, dims=dims, breakdown=breakdown)
                end
            end
            @testset "Vector Errors" begin
                dims = (4,)
                bad_breakdowns = [(1,), (1,2,), (3,)],     # can't have matrix
                                 hcat([(1,); (1,); (2,)])  # matrix shouldn't have vectors
                                 [(1,)]                    # dimensions mismatch
                for breakdown in bad_breakdowns
                    try
                        Jevo.WeightsCollection(rng, gene_counter, dims=dims, breakdown=breakdown)
                        @warn "Should have thrown an error for breakdown: $breakdown"
                        @test false 
                    catch
                        @test true
                    end
                end
                valid_breakdowns = [(1,), (1,), (2,)],
                                   [(4,)]
                for breakdown in valid_breakdowns
                    Jevo.WeightsCollection(rng, gene_counter, dims=dims, breakdown=breakdown)
                end
            end
            weight_collection = Jevo.WeightsCollection(rng, gene_counter, dims=(784, 15), breakdown=[(784,10) (784, 5);])
            weights = Jevo.tensor(weight_collection; weight_cache=weight_cache)
            @test size(weights) == (784, 15)
            @test !any(iszero, weights)

            visualize(weight_collection) # |> println
            # Test different breakdowns
            weight_collection = Jevo.WeightsCollection(rng, gene_counter, dims=(784, 15), breakdown=[(384,10) (384, 5); (400,10) (400, 5)])
            weights = Jevo.tensor(weight_collection; weight_cache=weight_cache)
            @test size(weights) == (784, 15)
            @test !any(iszero, weights)

        end
        # Test constructing with weight cache
        @testset "weight cache" begin
            push!(dense.weights.muts, NetworkGene(9000,9000, 0.1, Jevo.apply_kaiming_normal_noise!))
            push!(dense.bias.muts, NetworkGene(9001,9001, 0.1, Jevo.apply_kaiming_normal_noise!))
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
    end
    # Test phenotype creation & forward pass
    @testset "develop & forward pass full rank" begin
        state = State()
        gene_counter = Jevo.get_counter(AbstractGene, state)
        net = Jevo.Chain(rng, gene_counter, [(Jevo.Dense, (dims=(784,10), σ=relu))])
        dense = net.layers[1]
        creator = Creator(Flux.Chain)
        model = develop(creator, net)
        @test rand(Float32, 784) |> model |> size == (10,)
        # confirm we can get a list of weights 
        @test length(Jevo.get_weights(net)) == 2
        # Add mutations to each network
        @test all(map(w ->length(w.muts)==1, Jevo.get_weights(net, no_layer_norm=true)))
        mutated_net = Jevo.mutate(rng, state, nul_pop, net, mr=Float32(0.01))
        @test all(map(w ->length(w.muts)>=1, Jevo.get_weights(mutated_net, no_layer_norm=true)))
        @test all([w1.muts !== w2.muts for (w1, w2) in zip(Jevo.get_weights(net), Jevo.get_weights(mutated_net))])
        # Confirm developed weights are different after mutation
        model2 = develop(creator, mutated_net)
        @test model.layers[1].weight != model2.layers[1].weight
        @test model.layers[1].bias != model2.layers[1].bias
        # confirm that developing the same model twice results in the same layers
        model3 = develop(creator, net)
        @test model.layers[1].weight == model3.layers[1].weight
        @test model.layers[1].bias == model3.layers[1].bias
        # run multiple mutations, confirm we get different weights each time
        tmp_mutated_net = mutated_net
        prev_weight = model2.layers[1].weight
        prev_bias = model2.layers[1].bias
        for i in 1:10
            tmp_mutated_net = Jevo.mutate(rng, state, nul_pop, tmp_mutated_net, mr=Float32(0.01))
            model4 = develop(creator, tmp_mutated_net)
            @test model4.layers[1].weight != prev_weight
            @test model4.layers[1].bias != prev_bias
            prev_weight = model4.layers[1].weight
            prev_bias = model4.layers[1].bias
        end

        # even after creating new networks via mutation, we can still reconstruct the same ancestor
        model2_again = develop(creator, mutated_net)
        @test model2.layers[1].weight == model2_again.layers[1].weight
        @test model2.layers[1].bias == model2_again.layers[1].bias
    end
    @testset "low rank develop + fwd" begin
        state = State()
        gene_counter = Jevo.get_counter(AbstractGene, state)
        Main.weight_cache = WeightCache(maxsize=1_000_000)
        creator = Creator(Flux.Chain)
        full_net = Jevo.Chain(rng, gene_counter, [(Jevo.Dense, (dims=(784,100), σ=relu))])
        full_model = develop(creator, full_net)
        recon_full_net = Jevo.Chain(rng, gene_counter, [(Jevo.Dense, (dims=(784,100), σ=relu, rank=100))])
        recon_full_model = develop(creator, recon_full_net)
        lora_net = Jevo.Chain(rng, gene_counter, [(Jevo.Dense, (dims=(784,100), σ=relu, rank=32))])
        println(lora_net.layers[1])
        Main.weight_cache = WeightCache(maxsize=1_000_000)
        lora_model = develop(creator, lora_net)
        lora_model2 = develop(creator, lora_net)
        @test length(weight_cache) == 0 # don't add parent
        # confirm developing the same model twice results in the same layers
        @test lora_model.layers[1].weight == lora_model2.layers[1].weight
        @test lora_model.layers[1].bias == lora_model2.layers[1].bias
    
        @test rand(Float32, 784) |> full_model |> size == (100,)
        @test rand(Float32, 784) |> recon_full_model |> size == (100,)
        @test rand(Float32, 784) |> lora_model |> size == (100,)
    
        dense = full_model.layers[1]
        f_m, f_std = mean(dense.weight), std(dense.weight)
        dense = recon_full_model.layers[1]
        r_m, r_std = mean(dense.weight), std(dense.weight)
        dense = lora_model.layers[1]
        lora_m, lora_std = mean(dense.weight), std(dense.weight)
    
        @test r_m ≈ f_m atol=0.01
        @test r_std ≈ f_std atol=0.01
        @test lora_m ≈ f_m atol=0.01
        @test lora_std ≈ f_std atol=0.01
    
    end
    @testset "RNN" begin
        state = State()
        gene_counter = Jevo.get_counter(AbstractGene, state)
        Main.weight_cache = WeightCache(maxsize=1_000_000)
        n_blocks, n_heads, head_dim, hidden_dim, ff_dim, startsym, endsym, unksym, labels, seq_len = 2, 2, 5, 10, 20, "<s>", "</s>", "<unk>", string.(1:5), 8
        vocab = [unksym, startsym, endsym, labels...]
        vocab_size = length(vocab)

        textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)
        rnn_args = (hidden_dim = hidden_dim, vocab_size=vocab_size, σ=relu)
        rnn = TextRNN(rng, gene_counter, rnn_args)
        developer = Creator(TextModel, (;textenc=textenc))
        textrnn = develop(developer, rnn)
        env = RepeatSequence(n_labels=length(labels),
                             seq_len=8,
                             batch_size=7,
                             n_repeat=3)
        step!(env, [1], [textrnn])
        dummy_batch = rand(Float32, hidden_dim, seq_len, 7) |> Flux.gpu

        textrnn_jevo = develop(developer, rnn)
        textrnn_manual = develop(developer, rnn)
        # check params are same but objs are different
        @test textrnn_jevo != textrnn_manual
        @test Flux.params(textrnn_jevo) |> collect |> Iterators.flatten |> collect ==
                Flux.params(textrnn_manual) |> collect |> Iterators.flatten |> collect
        logits_jevo = Jevo.process_text_embeds(textrnn_jevo.model, dummy_batch, nothing)
        logits_manual = [textrnn_manual.model(dummy_batch[:,i,:]) for i in 1:seq_len]
        for i in 1:seq_len
            @test size(logits_jevo[:, i, :]) == size(logits_manual[i])
            @test logits_jevo[:, i, :] == logits_manual[i]
        end


    end
    @testset "Transformer" begin
        state = State()
        gene_counter = Jevo.get_counter(AbstractGene, state)
        Main.weight_cache = WeightCache(maxsize=1_000_000)
        n_blocks, n_heads, head_dim, hidden_dim, ff_dim, startsym, endsym, unksym, labels = 2, 2, 5, 10, 20, "<s>", "</s>", "<unk>", string.(1:5)
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
        jsa = Jevo.JevoSelfAttention(rng, gene_counter; attn_args...) 
        pnr_sa = Jevo.PostNormResidual(rng, gene_counter, sa; hidden_dim=hidden_dim)
        db = Jevo.TransformerDecoderBlock(rng, gene_counter; block_args...)
        trf = Jevo.TextTransformer(rng, gene_counter, tfr_args)
        visualize(trf) # make sure it doesn't error
        net = Jevo.TextTransformer(rng, gene_counter, tfr_args)
        weights = Jevo.get_weights(net)
        dims = [w.dims for w in weights]
        @test (hidden_dim,vocab_size) in dims # embed
        @test (vocab_size,) in dims # embed bias
        # make sure (10,5) is in dims exactly once, we don't want to
        # double count when sharing weights for embed/embeddecoder
        @test sum([d == (hidden_dim,vocab_size) for d in dims]) == 1
        #= # disabled, because we disabled feedforward weights =#
        #= @test (hidden_dim,ff_dim) in dims =#
        #= @test (ff_dim,hidden_dim) in dims =#
        # commented out because weights collection does not meet these dims
        # @test (3*head_dim*n_heads, hidden_dim) in dims # qkv
        @test (head_dim*n_heads, hidden_dim) in dims   # out
        @test (hidden_dim,) in dims # layernorm
        mutated_net = Jevo.mutate(rng, state, nul_pop, net, mr=Float32(0.01))
        Jevo.create_layer(embed; weight_cache=weight_cache)
        Jevo.create_layer(embed_decoder; weight_cache=weight_cache)
        Jevo.create_layer(sa; weight_cache=weight_cache)
        Jevo.create_layer(pnr_sa; weight_cache=weight_cache)
        Jevo.create_layer(db; weight_cache=weight_cache)
        Jevo.create_layer([db,db]; weight_cache=weight_cache)
    
        textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)
        creator = Creator(TextModel, (;textenc=textenc))
        trf_p = develop(creator, net) |> gpu
        seq = "1 2 1"
        one_batch_seq = batched([(seq,)])[1]
        input = encode(trf_p.textenc, one_batch_seq) |> gpu
        logits = trf_p(input)
        @test size(logits) == (8, 5, 1)
        # batching & masking
        sampled_batch = batched([(seq,) for i in 1:100])[1]
        input_batch = encode(trf_p.textenc, sampled_batch) |> gpu
        logits = trf_p(input_batch)
        @test size(logits) == (8, 5, 100)
        env = RepeatSequence(n_labels=length(labels),
                             seq_len=8,
                             batch_size=7,
                             n_repeat=3)
        @test length(Jevo.step!(env, [1], [trf_p])) == 1
        @test length(Jevo.play(env,[1], [trf_p])) == 1
        seq, logits = infer(trf_p, "1 2 1")
        @testset "LowRank" begin
            # LowRank
            net = Jevo.Chain(rng, gene_counter, [(Jevo.Dense, (dims=(784,10), σ=relu, rank=2))])
            dense = net.layers[1]
            d = Jevo.create_layer(dense; weight_cache=weight_cache)
            @test d.weight |> size == (10, 784)
            @test d.weight |> typeof == Array{Float32,2}
            @test d.bias |> size == (10,)
            @test d.bias |> typeof == Array{Float32,1}
            lora_tfr_args = (tfr_args..., qkv_rank=2, o_rank=2, ff_rank=2, embed_rank=2)
            net = Jevo.TextTransformer(rng, gene_counter, lora_tfr_args)
            visualize(net) |> println
            lora_tfr_p = develop(Creator(TextModel, (;textenc=textenc)), net)
        end

        @testset "HierarchicalTransformerTraverse" begin
            net = Jevo.TextTransformer(rng, gene_counter, tfr_args)
            # Check that we can use map to get all the weights of a network, should probably confirm
            # that ALL weights are retrieved, but that's probably another 20+ mins
            function get_n_muts(net)
                map(net, weights_only=true) do layers
                    length(layers[end].muts)
                end
            end
            n_muts = get_n_muts(net)
            @test length(n_muts) > 0 && all(n_muts .== 1)
            # Check that we can modify all the weights using map!
            map!(net, weights_only=true) do layers
                push!(layers[end].muts, NetworkGene(0,0,1f0,zero))
            end
            n_muts = get_n_muts(net)
            @test length(n_muts) > 0 && all(n_muts .== 2)
            # Confirm that mutation adds genes
            nul_pop = Population("", Individual[])
            mutated_net = mutate(rng, state, nul_pop, net, mr=0.1f0)

            #= visualize(mutated_net) |> println =#
        end
        @testset "Embed and Embed Decoder use same params" begin
            # confirm embed and embed decoder are the same after
            # 1. developing the network twice
            # 2. mutating the network
            # 3. developing the mutated network
            # 4. add delta to genome
            # 5. Base.+
            function check_same_geno_embeds(net::Jevo.TextNetwork)
                hash(net.embed) == hash(net.embeddecoder.embed)
            end
            function check_same_pheno_embeds(tm::TextModel)
                e_params = tm.embed |> cpu |> Flux.params |> Iterators.flatten |> collect
                ede_params = tm.embeddecoder.embed |> cpu |> Flux.params |> Iterators.flatten |> collect
                e_params == ede_params
            end
            net = Jevo.TextTransformer(rng, gene_counter, tfr_args)
            developer = Creator(TextModel, (;textenc=textenc))
            model1 = develop(developer, net)
            model2 = develop(developer, net)
            @test check_same_pheno_embeds(model2)  # 1.

            nul_pop = Population("", Individual[])
            mutated_net = mutate(rng, state, nul_pop, net, mr=0.1f0)
            @test check_same_geno_embeds(mutated_net)  # 2.

            model1 = develop(developer, mutated_net)
            @test check_same_geno_embeds(mutated_net)  # 3.

            delta = Delta(Jevo.TextTransformer(rng, gene_counter, tfr_args))
            @test check_same_geno_embeds(net + delta)  # 4.
            @test check_same_pheno_embeds(develop(developer, net + delta))  # 4.

            @test check_same_geno_embeds(Jevo.add_delta_to_genome(net, delta))  # 5.
            @test check_same_pheno_embeds(develop(developer, Jevo.add_delta_to_genome(net, delta)))  # 5.

        end
    end
end
