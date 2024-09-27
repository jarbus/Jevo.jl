export visualize, get_weights


# We need to overwrite this Flux method to generate Float32 weights and maintain compatibility with the (rng, type, dims...) signature
function kaiming_normal(rng::AbstractRNG,::Type, dims::Integer...; gain::Real = √2f0)
  std = Float32(gain / sqrt(first(Flux.nfan(dims...)))) # fan_in
  return randn(rng, Float32, dims...) .* std
end

# initialize factors with 2^(1/4) gain so when multiplied together,
# the resulting matrix has 2^(1/2) gain
apply_kaiming_normal_noise_factored!(rng::AbstractRNG, ::Type, arr::Array{Float32}, mr::Float32) =
    apply_kaiming_normal_noise!(rng, Float32, arr, mr, gain=2^(1/4))
    
function apply_kaiming_normal_noise!(rng::AbstractRNG, ::Type, arr::Array{Float32}, mr::Float32; gain::Real = √2f0)
    dims = size(arr)
    std = Float32(gain / sqrt(first(Flux.nfan(dims...))))
    scalar = std * mr
    @fastmath @inbounds @simd for i in 1:length(arr)
        arr[i] += randn(rng, Float32) * scalar
    end
end

function apply_gaussian_normal_noise!(rng::AbstractRNG, ::Type, arr::Array{Float32}, mr::Float32)
    @fastmath @inbounds @simd for i in 1:length(arr)
        arr[i] += randn(rng, Float32) * mr
    end
end

function apply_sparse_noise!(rng::AbstractRNG, ::Type, arr::Array{Float32}, mr::Float32)
    # choose a random number of elements to mutate
    n_elements = rand(rng, 1:length(arr))
    @inbounds for _ in 1:n_elements
        arr[rand(rng, 1:length(arr))] += randn(rng, Float32) * mr
    end
end

apply_zero!(rng::AbstractRNG, t::Type, arr::Array{Float32}, ::Float32) =
    apply_constant!(rng, t, arr, 0f0)

    apply_one!(rng::AbstractRNG, t::Type, arr::Array{Float32}, ::Float32) =
    apply_constant!(rng, t, arr, 1f0)

function apply_constant!(rng::AbstractRNG, ::Type, arr::Array{Float32}, v::Float32)
    @fastmath @inbounds @simd for i in 1:length(arr)
        arr[i] += v
    end
end

function get_weight_cache()
    # get global variable Main.weight_cache for weight cache
    # check if weight_cache is defined
    if !isdefined(Main, :weight_cache) || isnothing(Main.weight_cache)
        @warn "No weight cache found. Creating weight cache on proc $(myid())"
        Main.weight_cache = WeightCache(maxsize=300)
    end
    Main.weight_cache
end

function get_genotype_cache()
    # get global variable Main.weight_cache for weight cache
    # check if weight_cache is defined
    if !isdefined(Main, :genotype_cache) || isnothing(Main.genotype_cache)
        @warn "No genotype cache found. Creating genotype cache on proc $(myid())"
        Main.genotype_cache = GenotypeCache(maxsize=10)
    end
    Main.genotype_cache
end

function mr_symbol(mr::Float32)
    mr == 1.0f0 && return "#"  
    mr >= 0.1f0 && return "0"
    mr >= 0.01f0 && return "8"
    mr >= 0.001f0 && return "O"
    mr >= 0.0001f0 && return "1"
    mr >= 0.00001f0 && return "o"
    mr >= 0.000001f0 && return "."
    mr >= 0.0000001f0 && return "_"
    mr == 0.0f0 && return " "
    @error "mr too small to visualize"
end

function gene_symbol(prev_gene::NetworkGene, gene::NetworkGene)
    if gene.seed == prev_gene.seed
        if gene.mr == prev_gene.mr
            return "─"
        elseif gene.mr > prev_gene.mr
            return "<"
        else
            return ">"
        end
    elseif gene.init! == Jevo.apply_sparse_noise!
        gene.mr >= 0.001f0 && return "S"
        gene.mr < 0.001f0 && return "s"
    else
        return mr_symbol(gene.mr)
    end
end

function get_symbols(genes::Vector{NetworkGene})
    length(genes) == 0 && return ""
    symbols = String[mr_symbol(genes[1].mr)]
    for i in 2:length(genes)
        push!(symbols, gene_symbol(genes[i-1], genes[i]))
    end
    return join(symbols)
end

function get_weight_symbols(weights::Weights)
    str = lpad(string(weights.dims), 15) * " "
    str *= get_symbols(weights.muts) * "\n"
end

get_weight_symbols(::Nothing) = ""
get_weight_symbols(wc::WeightsCollection) = "weightscollection\n" *
    join([get_weight_symbols(w) for w in wc.weights])
get_weight_symbols(factorized_weights::FactorWeight) =
    get_weight_symbols(factorized_weights.A) * get_weight_symbols(factorized_weights.B)
get_weight_symbols(composite_weights::CompositeWeight) =
    join([get_weight_symbols(w) for w in composite_weights.weights])
get_weight_symbols(pnr::PostNormResidual) = get_weight_symbols(pnr.layer) * get_weight_symbols(pnr.norm)
get_weight_symbols(ln::LayerNorm) = "layernorm\n" * get_weight_symbols(ln.scale) * get_weight_symbols(ln.bias)
get_weight_symbols(sa::Union{JevoSelfAttention,SelfAttention}) =
    "qkv\n" * get_weight_symbols(sa.qkv) *
    "out\n" * get_weight_symbols(sa.out)
get_weight_symbols(d::Dense) =
    get_weight_symbols(d.weights) * get_weight_symbols(d.bias)
get_weight_symbols(e::Embed) = get_weight_symbols(e.weights)
get_weight_symbols(e::EmbedDecoder) =
    get_weight_symbols(e.embed) * get_weight_symbols(e.bias)
get_weight_symbols(network::JevoChain) = "chain\n"*join([get_weight_symbols(l) for l in network.layers])
get_weight_symbols(tfr::Transformer) = join([get_weight_symbols(l) for l in tfr.blocks])
    get_weight_symbols(rnn::Jevo.RNN) = get_weight_symbols(rnn.input)* get_weight_symbols(rnn.hidden) * get_weight_symbols(rnn.bias)
get_weight_symbols(tdb::TransformerDecoderBlock) =
    get_weight_symbols(tdb.attention) * get_weight_symbols(tdb.ff)

get_weight_symbols(tn::TextNetwork) = "TextNetwork\n" *
    "embed\n"* get_weight_symbols(tn.embed) *
    "network\n" * get_weight_symbols(tn.network) *
    "embeddecoder\n" * get_weight_symbols(tn.embeddecoder) 

visualize = get_weight_symbols
get_weight_symbols(ind::Individual{G,D,I}, pop::Population) where {G <: Delta,D,I} = get_weight_symbols(master_construct_genome(ind, pop))
get_weight_symbols(ind::Individual) = get_weight_symbols(ind.genotype)

is_layer_norm(layers) = any(l->l isa LayerNorm, layers)

function map_get(x::Union{AbstractLayer,AbstractWeights,AbstractGenotype}, type::Type)
    map(x) do hierarchy
        if hierarchy[end] isa type
            return hierarchy[end]
        end 
    end |> filter(!isnothing)
end

function get_weights(x::Union{AbstractLayer, AbstractGenotype, AbstractWeights}; no_layer_norm::Bool=false)
    map(x, weights_only=true) do hierarchy
        no_layer_norm && is_layer_norm(hierarchy) && return nothing
        return hierarchy[end]
    end |> filter(!isnothing)
end


function get_coupled_weights(x::Union{AbstractLayer, AbstractGenotype, AbstractWeights}; no_layer_norm::Bool=false)
    map(x, weights_only=false) do hierarchy
        no_layer_norm && is_layer_norm(hierarchy) && return nothing
        # treat as normal get_weights if we aren't in an attention block
        if !any(l->l isa JevoSelfAttention, hierarchy)
            return hierarchy[end] isa Weights ? hierarchy[end] : nothing
        end
        # If we are looking at sublayers of the attention block, skip, since we are coupling those
        !isa(hierarchy[end], JevoSelfAttention) && return nothing
        jsa = hierarchy[end]
        coupled_weights = []
        for i in 1:jsa.n_heads
            # *.weights and *.bias are WeightsCollections, so we need to get the weights 
            # of the collection
            
            # Group QK weights and biases
            qk = CoupledWeights([jsa.qkv.weights.weights[i], jsa.qkv.weights.weights[i+jsa.n_heads],
                                 jsa.qkv.bias.weights[i], jsa.qkv.bias.weights[i+jsa.n_heads]])
            # Group VO weights and biases
            vo = CoupledWeights([jsa.qkv.weights.weights[i+2jsa.n_heads], jsa.qkv.bias.weights[i+2jsa.n_heads],
                                 jsa.out.weights.weights[i]])
            # the out bias is (hidden_dim,) and there is only one per layer
            # so it is not coupled with the heads
            push!(coupled_weights, qk, vo, jsa.out.bias)
        end
        coupled_weights
    end |> filter(!isnothing) |> x->vcat(x...) # flatten the vectors of coupled weights
end

function set_device()
    worker_idx = myid() in workers() ? findfirst(x->x==myid(), sort(workers())) : 1
    println(" worker_idx=$worker_idx myid=$(myid()) workers=$(workers())")
    device_id = collect(devices())[worker_idx].handle |> Int64
    println("$(myid()) has devices $(collect(devices())), setting to $device_id")
    Main.jevo_device_id = device_id
    nothing
end

"""
    is_fresh(layer::AbstractLayer) -> Bool
    is_fresh(weight::AbstractWeights) -> Bool

Returns true if all weights in the layer are fresh, i.e. have only one gene with id < 0. Doesn't include weights that were created by population initializers.
"""
is_fresh(w::Weights) = length(w.muts) == 1 && w.muts[1].id < 0
is_fresh(layer::Union{AbstractLayer,AbstractWeights}) = all(is_fresh, get_weights(layer))


function samearchitecture(a, b)
    ws_a, ws_b = get_weights(a), get_weights(b)
    length(ws_a) != length(ws_b) && return false
    for (wa, wb) in zip(ws_a, ws_b)
        wa.dims != wb.dims && return false
    end
    true
end
