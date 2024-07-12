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
        Network([Jevo.Dense(Jevo.Weights((784,784),NetworkGene[]), Jevo.Weights((784,10),NetworkGene[]), relu)])
        # Better interface
        net = Network(rng, gene_counter, [(Jevo.Dense, (dims=(784,10), σ=relu))])
        dense = net.layers[1]

        @testset "tensor()" begin
            @test (10,784) == size(Jevo.tensor(dense.weights))
            @test mean(Jevo.tensor(dense.weights)) ≈ 0.0 atol=0.01
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
        net = Network(rng, gene_counter, [(Jevo.Dense, (dims=(784,10), σ=relu))])
        dense = net.layers[1]
        creator = Creator(Model)
        model = develop(creator, net)
        @test model |> typeof <: Model 
        @test rand(Float32, 784) |> model.chain |> size == (10,)
        # confirm we can get a list of weights 
        @test length(Jevo.get_weights(net)) == 2
        # Add mutations to each network
        @test all(map(w ->length(w.muts)==1, Jevo.get_weights(net)))
        mutated_net = Jevo.mutate(rng, state, nul_pop, net, mr=Float32(0.01))
        @test all(map(w ->length(w.muts)==1, Jevo.get_weights(mutated_net)))
        @test all([w1.muts !== w2.muts for (w1, w2) in zip(Jevo.get_weights(net), Jevo.get_weights(mutated_net))])
        # Confirm developed weights are different after mutation
        model2 = develop(creator, mutated_net)
        @test model.chain.layers[1].weight != model2.chain.layers[1].weight
        @test model.chain.layers[1].bias != model2.chain.layers[1].bias
        # confirm that developing the same model twice results in the same layers
        model3 = develop(creator, net)
        @test model.chain.layers[1].weight == model3.chain.layers[1].weight
        @test model.chain.layers[1].bias == model3.chain.layers[1].bias
        # run multiple mutations, confirm we get different weights each time
        tmp_mutated_net = mutated_net
        prev_weight = model2.chain.layers[1].weight
        prev_bias = model2.chain.layers[1].bias
        for i in 1:10
            tmp_mutated_net = Jevo.mutate(rng, state, nul_pop, tmp_mutated_net, mr=Float32(0.01))
            model4 = develop(creator, tmp_mutated_net)
            @test model4.chain.layers[1].weight != prev_weight
            @test model4.chain.layers[1].bias != prev_bias
            prev_weight = model4.chain.layers[1].weight
            prev_bias = model4.chain.layers[1].bias
        end

        # even after creating new networks via mutation, we can still reconstruct the same ancestor
        model2_again = develop(creator, mutated_net)
        @test model2.chain.layers[1].weight == model2_again.chain.layers[1].weight
        @test model2.chain.layers[1].bias == model2_again.chain.layers[1].bias
    end
    @testset "low rank develop + fwd" begin
        state = State()
        gene_counter = Jevo.get_counter(AbstractGene, state)
        Main.weight_cache = WeightCache(maxsize=1_000_000)
        creator = Creator(Model)
        full_net = Network(rng, gene_counter, [(Jevo.Dense, (dims=(784,100), σ=relu))])
        full_model = develop(creator, full_net)
        recon_full_net = Network(rng, gene_counter, [(Jevo.Dense, (dims=(784,100), σ=relu, rank=100))])
        recon_full_model = develop(creator, recon_full_net)
        lora_net = Network(rng, gene_counter, [(Jevo.Dense, (dims=(784,100), σ=relu, rank=32))])
        println(lora_net.layers[1])
        Main.weight_cache = WeightCache(maxsize=1_000_000)
        lora_model = develop(creator, lora_net)
        lora_model2 = develop(creator, lora_net)
        @test length(weight_cache) == 0 # don't add parent
        # confirm developing the same model twice results in the same layers
        @test lora_model.chain.layers[1].weight == lora_model2.chain.layers[1].weight
        @test lora_model.chain.layers[1].bias == lora_model2.chain.layers[1].bias
    
        @test rand(Float32, 784) |> full_model.chain |> size == (100,)
        @test rand(Float32, 784) |> recon_full_model.chain |> size == (100,)
        @test rand(Float32, 784) |> lora_model.chain |> size == (100,)
    
        dense = full_model.chain.layers[1]
        f_m, f_std = mean(dense.weight), std(dense.weight)
        dense = recon_full_model.chain.layers[1]
        r_m, r_std = mean(dense.weight), std(dense.weight)
        dense = lora_model.chain.layers[1]
        lora_m, lora_std = mean(dense.weight), std(dense.weight)
    
        @test r_m ≈ f_m atol=0.01
        @test r_std ≈ f_std atol=0.01
        @test lora_m ≈ f_m atol=0.01
        @test lora_std ≈ f_std atol=0.01
    
    end
    @testset "Transformer" begin
        state = State()
        gene_counter = Jevo.get_counter(AbstractGene, state)
        Main.weight_cache = WeightCache(maxsize=1_000_000)
        n_blocks = 2
        n_heads = 2
        head_dim = 5
        hidden_dim = 10
        ff_dim = 20
        startsym = "<s>"
        endsym = "</s>"
        unksym = "<unk>"
        labels = string.(1:5)
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
        pnr_sa = Jevo.PostNormResidual(rng, gene_counter, sa; hidden_dim=hidden_dim)
        db = Jevo.TransformerDecoderBlock(rng, gene_counter; block_args...)
        trf = Jevo.Transformer(rng, gene_counter; tfr_args...)
        visualize(trf) # make sure it doesn't error
        net = Network(rng, gene_counter, [(Jevo.Transformer, tfr_args)])
        weights = Jevo.get_weights(net)
        dims = [w.dims for w in weights]
        @test (hidden_dim,vocab_size) in dims # embed
        @test (vocab_size,) in dims # embed bias
        # make sure (10,5) is in dims exactly once, we don't want to
        # double count when sharing weights for embed/embeddecoder
        @test sum([d == (hidden_dim,vocab_size) for d in dims]) == 1
        @test (hidden_dim,ff_dim) in dims # ff
        @test (ff_dim,hidden_dim) in dims
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
        Jevo.create_layer((db,db); weight_cache=weight_cache)
    
        textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)
        creator = Creator(Jevo.TransformerPhenotype, (;textenc=textenc))
        trf_p = develop(creator, net) |> gpu
        seq = "1 2 1"
        one_batch_seq = batched([(seq,)])[1]
        input = preprocess(trf_p, one_batch_seq) |> gpu
        logits = trf_p(input)
        @test size(logits) == (8, 5, 1)
        # batching & masking
        sampled_batch = batched([(seq,) for i in 1:100])[1]
        input_batch = preprocess(trf_p, sampled_batch) |> gpu
        logits = trf_p(input_batch)
        @test size(logits) == (8, 5, 100)
        env = RepeatSequence(vocab_size=vocab_size,
                             seq_len=8,
                             batch_size=7,
                             n_repeat=3)
        @test length(Jevo.step!(env, [trf_p])) == 1
        @test length(Jevo.play(env, [trf_p])) == 1
        seq, logits = infer(trf_p, "1 2 1")
        # TODO TEST EXTENSIVELY
        @testset "LowRank" begin
            # LowRank
            net = Network(rng, gene_counter, [(Jevo.Dense, (dims=(784,10), σ=relu, rank=2))])
            dense = net.layers[1]
            d = Jevo.create_layer(dense; weight_cache=weight_cache)
            @test d.weight |> size == (10, 784)
            @test d.weight |> typeof == Array{Float32,2}
            @test d.bias |> size == (10,)
            @test d.bias |> typeof == Array{Float32,1}
            lora_tfr_args = (tfr_args..., qkv_rank=2, o_rank=2, ff_rank=2, embed_rank=2)
            net = Network(rng, gene_counter, [(Jevo.Transformer, lora_tfr_args)])
            visualize(net) |> println
            lora_tfr_p = develop(Creator(Jevo.TransformerPhenotype, (;textenc=textenc)), net)
        end
    end
end
