function get_binding(dims::NTuple{N, Int}, genes::Vector{NetworkGene}, idx::Int=length(genes))::WeightBinding where {N}
    length(genes) == 0 && error("No genes found")
    idx == 1 && return WeightBinding(dims, genes[1].seed, nothing)
    WeightBinding(dims, genes[idx].seed, genes[idx-1].seed)
end

############################
# PERFORMANCE CRITICAL START  (suspected)
function get_earliest_cached_weight(dims::NTuple{N, Int}, genes::Vector{NetworkGene}, weight_cache::_WeightCache)::Tuple{Array{Float32}, Int} where {N}
    """Return the earliest cached weight in the gene list. If none are cached, return a zero tensor of the given dimensions. Allocates memory. Also returns the idx of the earliest cached gene."""
    isnothing(weight_cache) && return zeros(Float32, dims), 0
    @inbounds @fastmath @simd for i in length(genes):-1:1
        binding = get_binding(dims, genes, i)
        weights = @inline get(weight_cache, binding, nothing)
        if !isnothing(weights)
            return copy(weights), i
        end
    end
    zeros(Float32, dims), 0
end
function tensor(w::Weights; weight_cache::_WeightCache=nothing)::Array{Float32}
    # ADD MUTS that have have not been cached
    dims, genes = w.dims, w.muts
    n_genes = length(genes)
    # get earliest cached weight or zero tensor if none found
    arr, ancestor_idx = @inline get_earliest_cached_weight(dims, genes, weight_cache)
    yes_weight_cache = !isnothing(weight_cache)
    # iteratively apply remaining mutations
    @inbounds @fastmath @simd for i in ancestor_idx+1:n_genes
        gene = genes[i]
        gid = gene.id
        rng = StableRNG(gene.seed)
        gene.init!(rng, Float32, arr, gene.mr)
        # update cache if we are using one
        if yes_weight_cache && i != n_genes && gid ∉ keys(weight_cache)
            binding = get_binding(dims, genes, i)
            weight_cache[binding] = copy(arr)
        end
    end
    arr
end
# PERFORMANCE CRITICAL END
############################

function create_layer(layer::Jevo.Dense; weight_cache::_WeightCache)
    weights = @inline tensor(layer.weights, weight_cache=weight_cache)
    bias = @inline tensor(layer.bias, weight_cache=weight_cache)
    Flux.Dense(weights, bias, layer.σ)
end
create_layer(f::Function; kwargs...) = f

function develop(::Creator{Model}, network::Network)
    # get global variable Main.weight_cache for weight cache
    # check if weight_cache is defined
    if !isdefined(Main, :weight_cache)
        @warn "No weight cache found, looking specifically for a variable called `weight_cache` in the global scope. Main has the following variables: $(names(Main))"
        Main.weight_cache = nothing
    end

    weight_cache::_WeightCache = Main.weight_cache
    Flux.Chain((create_layer(l, weight_cache=weight_cache) for l in network.layers)...) |> Model
end
