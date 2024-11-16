@testset "Tucker" begin
    state = State()
    gene_counter = Jevo.get_counter(AbstractGene, state)
    rng = StableRNG(1)

    #function Conv(rng::AbstractRNG, counter::AbstractCounter; kernel::Tuple{Vararg{Int}}, channels::Pair{<:Integer, <:Integer}, σ=relu, stride=(1,1), padding=(0,0, 0, 0), dilation=(1,1), groups=1, rank=(-1, -1, -1, -1))
    fura_g = Jevo.Conv(rng, gene_counter; kernel=(4,4), channels=32=>64, stride=(1,1), padding=(1,1, 1,1), dilation=(1,1), groups=1, rank=(-1, -1, -1, -1))
    fura_g_2 = Jevo.Conv(rng, gene_counter; kernel=(4,4), channels=32=>64, stride=(1,1), padding=(1,1, 1,1), dilation=(1,1), groups=1, rank=(-1, -1, -1, -1))
    lora_g = Jevo.Conv(rng, gene_counter; kernel=(4,4), channels=32=>64, stride=(1,1), padding=(1,1, 1,1), dilation=(1,1), groups=1, rank=4)
    fura_p = Jevo.create_layer(fura_g, weight_cache=weight_cache).weight
    lora_p = Jevo.create_layer(lora_g, weight_cache=weight_cache).weight
    # print means and std of weights
    println("fura_p.weights: mean=$(mean(fura_p)), std=$(std(fura_p)), max=$(maximum(fura_p)), min=$(minimum(fura_p))")
    println("lora_p.weights: mean=$(mean(lora_p)), std=$(std(lora_p)), max=$(maximum(lora_p)), min=$(minimum(lora_p))")

    fura_g = Jevo.Dense(rng, gene_counter; dims=(32,64), σ=relu)
    lora_g = Jevo.Dense(rng, gene_counter; dims=(32,64), σ=relu, rank=4)
    fura_p = Jevo.create_layer(fura_g, weight_cache=weight_cache).weight
    lora_p = Jevo.create_layer(lora_g, weight_cache=weight_cache).weight

    println("fura_p.weights: mean=$(mean(fura_p)), std=$(std(fura_p)), max=$(maximum(fura_p)), min=$(minimum(fura_p))")
    println("lora_p.weights: mean=$(mean(lora_p)), std=$(std(lora_p)), max=$(maximum(lora_p)), min=$(minimum(lora_p))")


end

