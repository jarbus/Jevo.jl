############################
# PERFORMANCE CRITICAL START  (suspected)
# possible optimizations: @inbounds, @fastmath
function get_earliest_cached_weight(dims::Tuple{Vararg{Int}}, genes::Vector{NetworkGene}, weight_cache::_WeightCache)
    """Return the earliest cached weight in the gene list. If none are cached, return a zero tensor of the given dimensions. Allocates memory. Also returns the idx of the earliest cached gene."""
    arr = zeros(Float16, dims)
    weight_cache === nothing && return arr, 0
    for i in length(genes):-1:1
        if genes[1:i] in keys(weight_cache)
            arr += weight_cache[genes[1:i]]
            return arr, i
        end
    end
    arr, 0
end
function tensor(w::Weights; weight_cache::_WeightCache=nothing)
    # ADD MUTS that have have not been cached
    dims, genes = w.dims, w.muts
    # get earliest cached weight or zero tensor if none found
    arr, ancestor_idx = get_earliest_cached_weight(dims, genes, weight_cache)
    yes_weight_cache = !isnothing(weight_cache)
    # iteratively apply remaining mutations
    for i in ancestor_idx+1:length(genes)
        gene = genes[i]
        arr .+= gene.mr .* gene.init(StableRNG(gene.seed), Float16, dims...)
        # update cache if we are using one
        yes_weight_cache && (weight_cache[genes[1:i]] = copy(arr))
    end
    arr
end
# PERFORMANCE CRITICAL END
############################

function create_layer(layer::Dense; weight_cache::_WeightCache)
    weights = tensor(layer.weights, weight_cache=weight_cache)
    bias = tensor(layer.bias, weight_cache=weight_cache)
    Flux.Dense(weights, bias, layer.σ)
end

function develop(::Creator, network::Network)
    # get global variable Main.weight_cache for weight cache
    # check if weight_cache is defined
    if !isdefined(Main, :weight_cache)
        @warn "No weight cache found, looking specifically for a variable called `weight_cache` in the global scope. Main has the following variables: $(names(Main))"
        Main.weight_cache = nothing
    end

    weight_cache = Main.weight_cache
    Chain(create_layer.(network.layers, weight_cache=weight_cache)...)
end


