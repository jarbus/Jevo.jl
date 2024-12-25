export visualize, get_weights


function cuda_randn(rng::AbstractRNG, dims::Integer...; std::Real=1.0f0)
    arr = randn(rng, Float32, dims...) #|> cu
    arr .*= std
    arr
end
# We need to overwrite this Flux method to generate Float32 weights and maintain compatibility with the (rng, type, dims...) signature
#= function kaiming_normal(rng::AbstractRNG,::Type, dims::Integer...; gain::Real = √2f0) =#
#=   std = Float32(gain / sqrt(first(Flux.nfan(dims...)))) # fan_in =#
#=   cuda_randn(rng, dims...; std=std) =#
#= end =#

function create_kaiming_nfactor_init(;n_factors::Int, dims=Tuple{Vararg{Int}})
    (rng, _, arr, mr) -> apply_kaiming_normal_noise_factored!(rng, Float32, arr, mr, n_factors=n_factors, dims=dims)
end

function apply_kaiming_normal_noise!(rng::AbstractRNG, ::Type, arr::Array{Float32}, mr::Float32; )
    apply_kaiming_normal_noise_factored!(rng, Float32, arr, mr, n_factors=1, dims=size(arr))
end

function apply_gaussian_normal_noise!(rng::AbstractRNG, ::Type, arr::Array{Float32}, mr::Float32)
    @fastmath @inbounds @simd for i in 1:length(arr)
        arr[i] += randn(rng, Float32) * mr
    end
end
    
function apply_kaiming_normal_noise_factored!(rng::AbstractRNG, ::Type, arr::Array{Float32}, mr::Float32; n_factors::Int, dims::Tuple{Vararg{Int}})
    if n_factors == 1
        n_i = first(Flux.nfan(size(arr)...))
        _std = sqrt(2/n_i)
    elseif ndims(arr) > 2 || dims[2] > dims[1]  # core matrix
        _std = Float32(sqrt(2/first(Flux.nfan(size(arr)...))))
    else  # factor matrix
        n_i = first(Flux.nfan(size(arr)...))
        _std = sqrt(1/n_i)
    end
    scalar = _std * mr
    arr .+= cuda_randn(rng, size(arr)...) .* scalar
end

#= function apply_gaussian_normal_noise!(rng::AbstractRNG, ::Type, arr::CuArray{Float32}, mr::Float32) =#
#=     arr .+= cuda_randn(rng, size(arr)...) .* mr =#
#= end =#

function apply_sparse_noise!(rng::AbstractRNG, ::Type, arr::CuArray{Float32}, mr::Float32)
    # choose a random number of elements to mutate
    n_elements = rand(rng, 1:length(arr))
    elements = rand(rng, 1:length(arr), n_elements)
    arr[elements] .+= cuda_randn(rng, n_elements) .* mr
end

apply_zero!(rng::AbstractRNG, t::Type, arr::Array{Float32}, ::Float32) = apply_constant!(rng, t, arr, 0f0)
apply_one!(rng::AbstractRNG, t::Type, arr::Array{Float32}, ::Float32) = apply_constant!(rng, t, arr, 1f0)
apply_constant!(::AbstractRNG, ::Type, arr::Array{Float32}, v::Float32) = (arr .+= v;)

global weight_cache = nothing
global genotype_cache = nothing


function get_weight_cache()
    # get global variable Jevo.weight_cache for weight cache
    # check if weight_cache is defined
    if !isdefined(Jevo, :weight_cache) || isnothing(Jevo.weight_cache)
        @warn "No weight cache found. Creating weight cache on proc $(myid())"
        Jevo.weight_cache = WeightCache(maxsize=Int(2^23), by=sizeof)
    end
    Jevo.weight_cache
end

function get_genotype_cache()
    # get global variable Jevo.weight_cache for weight cache
    # check if weight_cache is defined
    if !isdefined(Jevo, :genotype_cache) || isnothing(Jevo.genotype_cache)
        @warn "No genotype cache found. Creating genotype cache on proc $(myid())"
        Jevo.genotype_cache = GenotypeCache(maxsize=256)
    end
    Jevo.genotype_cache
end

function mr_symbol(mr::Float32)
    mr == 1.0f0 && return "#"  
    mr >= 0.2f0 && return "0"
    mr >= 0.1f0 && return "B"
    mr >= 0.02f0 && return "8"
    mr >= 0.01f0 && return "3"
    mr >= 0.001f0 && return "O"
    mr >= 0.0001f0 && return "1"
    mr >= 0.00001f0 && return "o"
    mr >= 0.000001f0 && return "."
    mr >= 0.0000001f0 && return "_"
    mr == 0.0f0 && return " "
    @error "mr too small to visualize"
end

function gene_symbol(gene::NetworkGene)
    if gene.init! == apply_zero!
        return " "
    elseif gene.init! == apply_one!
        return "|"
    else
        return mr_symbol(gene.mr)
    end
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
        return gene_symbol(gene)
    end
end

function get_symbols(genes::Vector{NetworkGene})
    length(genes) == 0 && return ""
    symbols = String[gene_symbol(genes[1])]
    for i in 2:length(genes)
        push!(symbols, gene_symbol(genes[i-1], genes[i]))
    end
    return join(symbols)
end

function get_weight_symbols(weights::Weights)
    str = lpad(string(weights.dims), 15) * " "
    str *= get_symbols(weights.muts) * "\n"
end

get_weight_symbols(::Union{Nothing, Function}) = ""
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
get_weight_symbols(l::Union{Dense, Conv}) = "dense\n" *
    get_weight_symbols(l.weights) * get_weight_symbols(l.bias)
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
function get_gpu_ids()
    gpu_ids = get(ENV, "SLURM_JOB_GPUS", "")
    if gpu_ids == ""
        println("No GPUs assigned to this job, defaulting to 0")
        return [0]
    end
    gpu_ids_array = split(gpu_ids, ",")
    return parse.(Int, gpu_ids_array)
end


function get_local_gpu_id()
    local_process_idx = parse(Int, get(ENV, "SLURM_LOCALID", "0"))
    gpu_ids = get_gpu_ids()
    # Ensure there are GPUs assigned
    if length(gpu_ids) == 0
        println("No GPUs assigned to this node.")
        return nothing
    end
    # Get the GPU ID assigned to the current process (wrap if more processes than GPUs)
    gpu_id = mod(local_process_idx, length(gpu_ids))
    println("local_process_idx $local_process_idx gpu_id=$gpu_id SLURM_JOB_GPUS=$(get(ENV, "SLURM_JOB_GPUS", "")) CUDA_VISIBLE_DEVICES=$(get(ENV, "CUDA_VISIBLE_DEVICES", ""))")
    return gpu_id
end

global jevo_device_id = nothing
function set_device()
    Jevo.jevo_device_id = get_local_gpu_id()
    @info "Setting device to $(Jevo.jevo_device_id)"
    device!(Jevo.jevo_device_id)
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

function align_weight_vectors!(genome::Vector, delta::Vector)
    @assert length(genome) <= length(delta) 
    for i in eachindex(delta)
        !is_fresh(delta[i]) && continue
        insert!(genome, i, nothing)
    end
end
