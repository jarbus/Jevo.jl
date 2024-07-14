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
    if !isdefined(Main, :weight_cache)
        @warn "No weight cache found. Creating weight cache on proc $(myid())"
        Main.weight_cache = WeightCache(maxsize=Int(1e8))
    end
    Main.weight_cache
end

function get_genotype_cache()
    # get global variable Main.weight_cache for weight cache
    # check if weight_cache is defined
    if !isdefined(Main, :genotype_cache)
        @warn "No genotype cache found. Creating genotype cache on proc $(myid())"
        Main.genotype_cache = GenotypeCache(maxsize=Int(1e8))
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
    else
        gene.init! == Jevo.apply_sparse_noise! && return "s"
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

get_weight_symbols(wc::WeightsCollection) = "weightscollection\n" *
    join([get_weight_symbols(w) for w in wc.weights])
get_weight_symbols(factorized_weights::FactorWeight) =
    get_weight_symbols(factorized_weights.A) * get_weight_symbols(factorized_weights.B)
get_weight_symbols(composite_weights::CompositeWeight) =
    join([get_weight_symbols(w) for w in composite_weights.weights])
get_weight_symbols(pnr::PostNormResidual) = get_weight_symbols(pnr.layer) * get_weight_symbols(pnr.norm)
get_weight_symbols(ln::LayerNorm) = "layernorm\n" * get_weight_symbols(ln.scale) * get_weight_symbols(ln.bias)
get_weight_symbols(sa::SelfAttention) =
    "qkv\n" * get_weight_symbols(sa.qkv) *
    "out\n" * get_weight_symbols(sa.out)
get_weight_symbols(d::Dense) =
    get_weight_symbols(d.weights) * get_weight_symbols(d.bias)
get_weight_symbols(e::Embed) = get_weight_symbols(e.weights)
get_weight_symbols(e::EmbedDecoder) =
    get_weight_symbols(e.embed) * get_weight_symbols(e.bias)
get_weight_symbols(c::Chain) = 
    "chain\n" * join([get_weight_symbols(l) for l in c.layers])
get_weight_symbols(tdb::TransformerDecoderBlock) =
    get_weight_symbols(tdb.attention) * get_weight_symbols(tdb.ff)

get_weight_symbols(t::Transformer) = "Transformer\n" *
    "embed\n"* get_weight_symbols(t.embed) *
    "blocks\n" * join([get_weight_symbols(b) for b in t.blocks]) *
    "embeddecoder\n" * get_weight_symbols(t.embeddecoder) 

get_weight_symbols(network::Network) = join([get_weight_symbols(l) for l in network.layers])

visualize = get_weight_symbols
get_weight_symbols(ind::Individual{G,D,I}, pop::Population) where {G <: Delta,D,I} = get_weight_symbols(master_construct_genome(ind, pop))
get_weight_symbols(ind::Individual) = get_weight_symbols(ind.genotype)

is_layer_norm(layers) = any(l->l isa LayerNorm, layers)

function get_weights(x::Union{Network, AbstractLayer, AbstractGenotype}; no_layer_norm::Bool=false)
    map(x, weights_only=true) do hierarchy
        no_layer_norm && is_layer_norm(hierarchy) && return nothing
        return hierarchy[end]
    end |> x->filter(!isnothing, x)
end

function set_device()
    worker_idx = myid() in workers() ? findfirst(x->x==myid(), sort(workers())) : 1
    println(" worker_idx=$worker_idx myid=$(myid()) workers=$(workers())")
    device_id = collect(devices())[worker_idx].handle |> Int64
    println("$(myid()) has devices $(collect(devices())), setting to $device_id")
    Main.jevo_device_id = device_id
    nothing
end
