export NNGenePoolMutator, ClearCurrentGenWeights, ComputeGenePoolNetwork, AddAttentionHeads, AddDecoderBlock

@define_op "ComputeGenePoolNetwork"
@define_op "ClearCurrentGenWeights" "AbstractMutator"
@define_op "NNGenePoolMutator" "AbstractMutator"
@define_op "AddAttentionHeads" "AbstractMutator"
@define_op "AddDecoderBlock" "AbstractMutator"


function compute_init(layers)
    n_factors = 0
    for layer in layers
        if layer isa FactorWeight
            n_factors += 1
        elseif layer isa LayerNorm && n_factors > 0
            strings = [string(typeof(layer)) for layer in layers]
            @error "LayerNorm not supported $strings"
        end
    end
    if n_factors == 0
        return apply_kaiming_normal_noise!
    elseif n_factors == 1
        return apply_kaiming_normal_noise_factored!
    else
        @error "$n_factors factors not supported"
    end
end
function what_layers_should_we_mutate(rng::AbstractRNG, genotype::Network; n::Int, no_layer_norm::Bool)
    n_weights = length(get_weights(genotype, no_layer_norm=no_layer_norm))
    n == -1 && return ones(Bool, n_weights)
    should_mutate = zeros(Bool, n_weights)
    should_mutate[1:n] .= true
    shuffle!(rng, should_mutate)
    should_mutate
end

"""
Uses the Mutation API to clear all weights of the current generation's individuals.

Designed to be used with [Deltas](@ref Delta) which have been cloned.
"""
ClearCurrentGenWeights(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, kwargs...) = 
    create_op("ClearCurrentGenWeights", 
              condition=condition,
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)->mutate!(s, p; fn=clear_weights, kwargs...))),
              time=time;)
clear_weights(::AbstractRNG, ::State, ::Population, genotype) =
    copyarchitecture(genotype)

function mutate(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Network; mr::Union{Float32,Tuple{Vararg{Float32}}}, n::Int=-1, no_layer_norm::Bool=true, kwargs...)
    genotype = deepcopy(genotype)
    gene_counter = get_counter(AbstractGene, state)
    # Choose weights to mutate
    should_mutate = what_layers_should_we_mutate(rng, genotype, n=n, no_layer_norm=no_layer_norm)
    # Determine which weights to mutate based off n
    map!(genotype, weights_only=true) do layers
        weight = layers[end]               
        no_layer_norm && is_layer_norm(layers) && return     # Skip if we're a layer norm
        !popfirst!(should_mutate) && return # Skip if we don't want to mutate this weight
        # init = rand(rng, (compute_init(layers), apply_sparse_noise!))
        init = compute_init(layers)
        mrf0 = mr isa Float32 ? mr : rand(rng, mr)
        push!(weight.muts, NetworkGene(rng, gene_counter, mrf0, init))
    end
    @assert isempty(should_mutate) "Should have iterated through all weights, $(length(should_mutate)) left"
    genotype
end

mutate(rng::AbstractRNG, state::State, population::AbstractPopulation, genotype::Delta, args...; kwargs...) =
    Delta(mutate(rng, state, population, genotype.change, args...; kwargs...))

"""
Network where each weight contains all genes in the gene pool
"""
struct GenePoolNetwork
    network
end
function update_genepool_network!(pop::Population; no_layer_norm::Bool)
    gp = getonly(x->x isa GenePool, pop.data)
    length(gp.deltas) == 0 && @error "Genepool empty" 
    stat_net = copyarchitecture(gp.deltas[1])
    stat_net_ws = Jevo.get_weights(stat_net, no_layer_norm=no_layer_norm)
    mut_found = false
    for d in gp.deltas
        for (stat_net_w, d_w) in zip(stat_net_ws, Jevo.get_weights(d, no_layer_norm=no_layer_norm))
            push!(stat_net_w.muts, d_w.muts...)
            mut_found = mut_found || !isempty(d_w.muts)
        end
    end
    @assert mut_found "No mutations found when creating GenePoolNetwork"
    filter!(d->!isa(d, GenePoolNetwork), pop.data) # get rid of any existing genepools
    push!(pop.data, GenePoolNetwork(stat_net))
end

ComputeGenePoolNetwork(ids::Vector{String}=String[];after_gen::Int, no_layer_norm::Bool=true, kwargs...) =
    create_op("ComputeGenePoolNetwork",
    condition=s->generation(s) > after_gen,
    retriever=PopulationRetriever(ids),
    updater=map(map((_,p)->update_genepool_network!(p, no_layer_norm=no_layer_norm))); kwargs...)

function nn_genepool_mutate(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Network, args...; mr::Union{Float32,Tuple{Vararg{Float32}}}, n::Int=-1, no_layer_norm::Bool=true, kwargs...)

    gp_network = getonly(d->d isa GenePoolNetwork, pop.data)
    gp_weights = Jevo.get_weights(gp_network.network, no_layer_norm=no_layer_norm)
    genotype = deepcopy(genotype)
    gene_counter = get_counter(AbstractGene, state)
    n_deltas = getonly(x->x isa GenePool, pop.data).deltas |> length
    n_sparse, n_muts = 0, 0
    for gp_w in gp_weights, mut in gp_w.muts
        n_sparse += mut.init! == apply_sparse_noise!
        n_muts += 1
    end
    percent_sparse = n_sparse / n_muts
    @assert 0 <= percent_sparse <= 1
    # entire mutation has to be either sparse or full, not on a per-gene basis
    if rand(rng) < 0.01  # sample random init with small chance
        sparse_mut = rand(rng, Bool)
    else 
        sparse_mut = rand(rng) < percent_sparse  # sample sparse mut with small chance
    end

    added_weight = false
    loops = 0
    while !added_weight
        should_mutate = what_layers_should_we_mutate(rng, genotype, n=n, no_layer_norm=no_layer_norm)
        # Determine which weights to mutate based off n
        idx = 0
        map!(genotype, weights_only=true) do layers
            weight = layers[end]               
            no_layer_norm && is_layer_norm(layers) && return     # Skip if we're a layer norm
            idx += 1
            gp_w = gp_weights[idx]
            !popfirst!(should_mutate) && return # Skip if we don't want to mutate this weight
            rand(rng, 1:n_deltas) > length(gp_w.muts) + 1 && return # skip proportional to selected weight
            mrf0 = mr isa Float32 ? mr : rand(rng, mr)
            init! = sparse_mut ? apply_sparse_noise! : compute_init(layers)
            max_mr = maximum(m.mr for m in gp_w.muts; init=0f0)
            max_mr = max_mr == 0f0 ? rand(rng, mr) : max_mr
            mrf0 > max_mr && rand(rng) > 0.01 && return
            gene = NetworkGene(rng, gene_counter, mrf0, init!) 
            push!(weight.muts, gene)
            added_weight = true
        end
        loops += 1
        @assert loops <= 10
        @assert idx == length(gp_weights) "Should have iterated through all weights, idx=$idx, $(length(gp_weights)) left"
        @assert isempty(should_mutate) "Should have iterated through all weights, $(length(should_mutate)) left"
    end
    genotype
end

nn_genepool_mutate(rng::AbstractRNG, state::State, population::AbstractPopulation, genotype::Delta, args...; kwargs...) =
    Delta(nn_genepool_mutate(rng, state, population, genotype.change, args...; kwargs...))

NNGenePoolMutator(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, kwargs...) = 
    create_op("NNGenePoolMutator", 
              condition=condition,
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)->mutate!(s, p; fn=nn_genepool_mutate, kwargs...))),
              time=time;)

""" Choose a random layer to add an attention head to."""
function add_attention_head(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Network, args...; prob::Float64,kwargs...)
    rand(rng) > prob && return genotype
    genotype = deepcopy(genotype)
    gene_counter = get_counter(AbstractGene, state)
    @assert genotype.layers[1] isa Transformer "Must be a Transformer"
    attention_layers = [l.attention.layer for l in genotype.layers[1].blocks]
    @assert length(attention_layers) > 0 "No attention layers found"
    attn_layer = rand(rng, attention_layers)
    weight_collections = get_weight_collections(attn_layer)
    # Find weight collection sub-layer. This needs to work with lora and non-lora.
    @assert length(weight_collections) > 0 "No weight collections found"
    wc = rand(rng, weight_collections)
    @assert wc.dims[1] % 3 == 0 "Found weight collection in attention layer not divisible by 3, got $(wc.dims[2])"
    @assert wc.weights[1] isa Weights "First weight in weight collection is not a Weights, behavior is undefined"
    # insert three heads, one for q, k, v
    dims = wc.weights[1].dims
    init! = wc.weights[1].muts[1].init!
    third = div(dims[1], 3)
    # Do this in reverse order so we don't mess up the indices
    for i in (3,2,1) 
      # insert at the end of each third of an arrayy
      # [ 1 2 3 ] heads [ a b c d e f ] weights third=2
      # [ 1 2 ] heads [ a b c d e f 3] weights
      # [ 1 ] heads [ a b c d 2 e f 3] weights
      # [ ] heads [ a b 1 c d 2 e f 3] weights
      head = Weights(dims, [NetworkGene(-inc!(gene_counter), rand(rng, UInt64), 1f0, apply_kaiming_normal_noise!)])
      insert!(wc.weights, (i*third)+1, head)
    end
    attn_layer.n_heads += 1
    @assert length(wc.weights) == 3 * attn_layer.n_heads "Invalid number of heads"
    genotype
end

add_attention_head(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Delta, args...; kwargs...) = 
    Delta(add_attention_head(rng, state, pop, genotype.change, args...; kwargs...))

AddAttentionHeads(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, kwargs...) = 
    create_op("AddAttentionHeads", 
              condition=condition,
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)-> mutate!(s, p; fn=add_attention_head, kwargs...))), time=time;)

function add_decoder_block(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Network, args...; prob::Float64, head_dims::Tuple{Vararg{Int}}, ff_dim::Int, hidden_dim::Int, qkv_rank::Int=-1, o_rank::Int=-1, ff_rank::Int=-1, kwargs...)
    rand(rng) > prob && return genotype
    genotype = deepcopy(genotype)
    gene_counter = get_counter(AbstractGene, state)
    @assert genotype.layers[1] isa Transformer "Must be a Transformer"
    head_dim = rand(rng, head_dims)
    # Create new block with one head
    block = TransformerDecoderBlock(rng, gene_counter, n_heads=1, head_dim=head_dim, hidden_dim=hidden_dim, ff_dim=ff_dim, qkv_rank=qkv_rank, o_rank=o_rank, ff_rank=ff_rank)
    # Invert ids of all weights in block to indicate new genes
    map!(block, weights_only=true) do layers
        muts = layers[end].muts
        @assert length(muts) == 1 "Expected one NetworkGene for $(layers)"
        gene = muts[1]
        muts[1] = NetworkGene(-gene.id, gene.seed, gene.mr, gene.init!)
    end
    # randomly insert the block
    blocks = genotype.layers[1].blocks
    insert!(blocks, rand(rng, 1:length(blocks)+1), block)
    @info visualize(genotype)
    genotype
end
add_decoder_block(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Delta, args...; kwargs...) = 
    Delta(add_decoder_block(rng, state, pop, genotype.change, args...; kwargs...))

AddDecoderBlock(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, kwargs...) = 
    create_op("AddDecoderBlock", 
              condition=condition,
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)-> mutate!(s, p; fn=add_decoder_block, kwargs...))), time=time;)
