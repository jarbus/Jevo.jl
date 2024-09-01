export ClearCurrentGenWeights, AddAttentionHeads, AddDecoderBlock, NBackMutator

@define_op "ClearCurrentGenWeights" "AbstractMutator"
@define_op "AddAttentionHeads" "AbstractMutator"
@define_op "AddDecoderBlock" "AbstractMutator"
@define_op "NBackMutator" "AbstractMutator"

"""
    non_ancestral_mutate!(rng, gene_counter, hist_weight::Weights, weight::Weights; mrs::Tuple{Vararg{Float32}})

Adds a new gene to the weight. If `hist_weight` is provided, the gene's initialization is that of the last historical mutation, otherwise, we assume the gene is fresh and use the only initialization in `weight.muts`.

Returns true, because a mutation is added.
"""
function non_ancestral_mutate!(rng, gene_counter, hist_weight::Weights, weight::Weights; mrs::Tuple{Vararg{Float32}})
    init! = hist_weight.muts[end].init!
    push!(weight.muts, NetworkGene(rng, gene_counter, rand(rng, mrs), init!))
    true
end

"""
    ancestral_mutate!(rng, gene_counter, historical_weight, weight::Weights; n_back::Int, mrs::Tuple{Vararg{Float32}})

Adds a new gene to the weight. If randomly sampled MR is greater than the max MR in the last `n_back` mutations, we skip the mutation.

Returns true if a mutation is added, false otherwise.
"""
function ancestral_mutate!(rng, gene_counter, historical_weight, weight::Weights; n_back::Int, mrs::Tuple{Vararg{Float32}})
    # choose whether to use existing or sparse init
    init! = if rand(rng) < 0.01  # sample random init with small chance
        rand(rng, (apply_kaiming_normal_noise!, apply_sparse_noise!))
    else  # otherwise, sample init based on previously selected inits
        rand(rng, historical_weight.muts[end-n_back+1:end]).init!
    end
    # choose a mutation rate. if it's higher than the max selected mutation rate, skip
    # with 0.01 chance, we can sample a higher mutation rate
    mr = rand(rng, mrs)
    max_mr = maximum(m.mr for m in historical_weight.muts)
    mr > max_mr && rand(rng) > 0.01 && return false
    gene = NetworkGene(rng, gene_counter, mr, init!) 
    push!(weight.muts, gene)
    return true
end

# TODO FIX this is a non-deterministic hack, because gene ids are assigned in threads
# We need to figure out a fix for this, for reproducibility
function compute_mutation_probabilities(weights)
    last_gene_ids = Float64.([abs(weight.muts[end].id) for weight in weights])
    last_gene_ids .= round.(last_gene_ids ./ 10000 ) # TODO hack to mitigate non-determinism, get rid of this
    last_gene_ids .-= minimum(last_gene_ids)
    _max = maximum(last_gene_ids)
    if _max != 0
        last_gene_ids ./= _max
    end
    last_gene_ids
end

function nback_mutate(rng::AbstractRNG, state::State, ::AbstractPopulation, ind::Individual; n_back::Int, min_mutation_prob::Float64=0.05, mrs::Tuple{Vararg{Float32}}, no_layer_norm::Bool=true, max_n_muts::Int)
    genome = deepcopy(ind.genotype)
    weights = get_weights(genome, no_layer_norm=no_layer_norm)
    historical_genome = ind.generation == 0 ? deepcopy(genome) : get_genotype_cache()[ind.parents[1]]
    gene_counter = @inline get_counter(AbstractGene, state)
    historical_weights = get_weights(historical_genome, no_layer_norm=no_layer_norm)
    probabilities = @inline compute_mutation_probabilities(historical_weights)
    @assert samearchitecture(historical_genome, genome)
    @assert length(weights) == length(historical_weights) == length(probabilities)
    tries, n_added_mutations = 0, 0
    while tries < 50 && n_added_mutations == 0
        random_order = zip(weights, historical_weights, probabilities) |> collect |> shuffle
        for (weight, hist_weight, prob) in random_order
            rand(rng) > prob - min_mutation_prob && continue
            added_mut = if length(hist_weight.muts) < n_back
                @inline non_ancestral_mutate!(rng, gene_counter, hist_weight, weight, mrs=mrs)
            else
                @inline ancestral_mutate!(rng, gene_counter, hist_weight, weight, mrs=mrs, n_back=n_back)
            end
            n_added_mutations += added_mut
            n_added_mutations == max_n_muts && break
        end
        n_added_mutations > 0 && break
        tries += 1
    end
    @assert n_added_mutations > 0 "No mutations added given probabilities $probabilities"
    genome
end

function mutate!(rng::AbstractRNG, state::AbstractState, population::AbstractPopulation, ind::Individual{G, D, I}, args...; fn, kwargs...) where {G <: Delta, D, I}
    # I really, really don't like this, but this is simpler less confusing than the alternative
    if fn != nback_mutate
        ind.genotype = fn(rng, state, population, ind.genotype, args...; kwargs...)
        return
    end
    ind.genotype = fn(rng, state, population, ind, args...; kwargs...)
end


"""
    NBackMutator(ids::Vector{String}=String[]; n_back::Int, mrs::Tuple{Vararg{Float32}}, no_layer_norm::Bool, condition::Function=always, time::Bool=false, min_mutation_prob::Float64=0.05, max_n_muts::Int)

Mutates up to `max_n_muts` weights of the current individual based on the last `n_back` mutations in the historical individual. If the historical individual has less than `n_back` mutations, we mutate randomly. The mutation rate of the new gene is sampled from `mrs`, and if the sampled mutation rate is higher than the maximum mutation rate in the last `n_back` mutations, we skip the mutation with a `min_mutation_prob` chance.
"""
NBackMutator(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, kwargs...) = 
    create_op("NBackMutator",
              condition=condition,
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)->mutate!(s, p; fn=nback_mutate, kwargs...))),
              time=time;)


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
    should_mutate[1:min(n,n_weights)] .= true
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
clear_weights(::AbstractRNG, ::State, ::Population, genotype) = copyarchitecture(genotype)


# Underpowered mutation op
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
        init = compute_init(layers)
        mrf0 = mr isa Float32 ? mr : rand(rng, mr)
        push!(weight.muts, NetworkGene(rng, gene_counter, mrf0, init))
    end
    @assert isempty(should_mutate) "Should have iterated through all weights, $(length(should_mutate)) left"
    genotype
end

mutate(rng::AbstractRNG, state::State, population::AbstractPopulation, genotype::Delta, args...; kwargs...) =
    Delta(mutate(rng, state, population, genotype.change, args...; kwargs...))

# Adds attention head to random self-attention layer
function add_attention_head(rng::AbstractRNG, state::State, ::AbstractPopulation, genotype::Network, args...; prob::Float64, inits::Tuple{Vararg{Function}}, kwargs...)
    @assert genotype.layers[1] isa Transformer "Must be a Transformer"
    rand(rng) > prob && return genotype
    genotype = deepcopy(genotype)
    gene_counter = get_counter(AbstractGene, state)
    # Get random weight collection within a random attention layer
    attention_layers = map_get(genotype, SelfAttention)
    @assert length(attention_layers) > 0 "No attention layers found"
    attn_layer = rand(rng, attention_layers) 
    weight_collections = map_get(attn_layer, WeightsCollection)
    # Find weight collection sub-layer. This needs to work with lora and non-lora.
    @assert length(weight_collections) != 0 "No weight collections found"
    @assert length(weight_collections) % 3 == 0 "There should be a multiple of 3 weight collects, qkv weight, qkv bias, out weight"
    @assert length(weight_collections) == 3 "Only one weight collection per matrix supported for dynamic heads right now"
    attn_layer.n_heads += 1
    init! = rand(rng, inits)
    for wc in weight_collections
        @assert length(size(wc.weights)) == 1 || 1 ∈ size(wc.weights) "WeightCollection should have one dimension or a dimension of length 1"
        @assert wc.weights[1] isa Weights "First weight in weight collection is not a Weights, behavior is undefined"
    end
    for wc in weight_collections[1:2]  # qkv weight and bias
        @assert length(wc.weights) % 3 == 0 "Found weight collection in attention layer not divisible by 3, got $(wc.dims[1])"
        # insert three heads, one for q, k, v
        dims = wc.weights[1].dims
        third = div(length(wc.weights), 3)
        # Do this in reverse order so we don't mess up the indices
        for i in (3,2,1) 
          # insert at the end of each third of an arrayy
          # [ 1 2 3 ] heads [ a b c d e f ] weights third=2
          # [ 1 2 ] heads [ a b c d e f 3] weights
          # [ 1 ] heads [ a b c d 2 e f 3] weights
          # [ ] heads [ a b 1 c d 2 e f 3] weights
          head = Weights(dims, [NetworkGene(-inc!(gene_counter), rand(rng, UInt64), 0.1f0, init!)])
          insert!(wc.weights, (i*third)+1, head)
        end
        @assert length(wc.weights) == 3 * attn_layer.n_heads "Invalid number of heads, got $(length(wc.weights)), expected $(3 * attn_layer.n_heads)"
    end
    wc = weight_collections[3]  # out projection weights
    dims = wc.weights[1].dims
    @assert size(wc.weights, 1) == 1 "Out projection weight collection should have a first dimension of 1"
    head = Weights(dims, [NetworkGene(-inc!(gene_counter), rand(rng, UInt64), 0.1f0, init!)])
    wc.weights = hcat(wc.weights, head)
    @assert size(wc.weights, 2) == attn_layer.n_heads "Invalid number of heads, got $(length(wc.weights)), expected $(attn_layer.n_heads)"
    update_dimensions!(genotype)
    genotype
end

add_attention_head(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Delta, args...; kwargs...) = 
    Delta(add_attention_head(rng, state, pop, genotype.change, args...; kwargs...))

"""
    AddAttentionHeads(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, prob::Float64, inits::Tuple{Vararg{Function}}, kwargs...)

Adds a new attention head to a random self-attention layer with a probability of `prob`. The new head is initialized with a randomly selected function from `inits`.
"""
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
    map!(block, weights_only=true) do hierarchy
        mr =is_layer_norm(hierarchy) ? 1f0 : 0.1f0
        muts = hierarchy[end].muts
        @assert length(muts) == 1 "Expected one NetworkGene for $(hierarchy)"
        gene = muts[1]
        muts[1] = NetworkGene(-gene.id, gene.seed, mr, gene.init!)
    end
    # randomly insert the block
    blocks = genotype.layers[1].blocks
    insert!(blocks, rand(rng, 1:length(blocks)+1), block)
    genotype
end
add_decoder_block(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Delta, args...; kwargs...) = 
    Delta(add_decoder_block(rng, state, pop, genotype.change, args...; kwargs...))

"""
    AddDecoderBlock(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, prob::Float64, head_dims::Tuple{Vararg{Int}}, ff_dim::Int, hidden_dim::Int, qkv_rank::Int=-1, o_rank::Int=-1, ff_rank::Int=-1, kwargs...)

Adds a new decoder block to the transformer with a probability of `prob`. The new block has one head with a dimension randomly selected from `head_dims`.
"""
AddDecoderBlock(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, kwargs...) = 
    create_op("AddDecoderBlock", 
              condition=condition,
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)-> mutate!(s, p; fn=add_decoder_block, kwargs...))), time=time;)
