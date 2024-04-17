function compute_init(layers)
    n_factors = 0
    for layer in layers
        if layer isa FactorWeight
            n_factors += 1
        elseif layer isa LayerNorm
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
function mutate_weights!(rng::AbstractRNG, state::State, genotype::Network, mr::Union{Float32,Tuple{Vararg{Float32}}}, lookback::Int; n=-1)
    gene_counter = get_counter(AbstractGene, state)
    # Choose weights to mutate

    # This is the easiest way to get the number of weights in a network,
    # doesn't require additional params, and works with dynamic networks
    # Counts all the weights that are not layer norms
    n_weights = length(get_weights(genotype, no_layer_norm=true))
    # Determine which weights to mutate based off n
    if n == -1
        should_mutate = ones(Bool, n_weights)
    else
        should_mutate = zeros(Bool, n_weights)
        should_mutate[1:n] .= true
        shuffle!(rng, should_mutate)
    end
    map!(genotype, weights_only=true) do layers
        weight = layers[end]               
        empty!(weight.muts)                 # First, clear copied mutations, since this is a delta
        is_layer_norm(layers) && return     # Skip if we're a layer norm
        !popfirst!(should_mutate) && return # Skip if we don't want to mutate this weight
        init = compute_init(layers)
        mrf0 = mr isa Float32 ? mr : rand(rng, mr)
        push!(weight.muts, NetworkGene(rng, gene_counter, mrf0, init))
    end
    @assert isempty(should_mutate) "Should have iterated through all weights, $(length(should_mutate)) left"
end

function mutate(rng::AbstractRNG, state::State, genotype::Network; mr::Union{Float32,Tuple{Vararg{Float32}}}, n::Int=2, lookback::Int=-1)
    new_genotype = deepcopy(genotype)
    gene_counter = get_counter(AbstractGene, state)
    mutate_weights!(rng, state, new_genotype, mr, lookback, n=n)
    new_genotype
end
mutate(rng::AbstractRNG, state::State, genotype::Delta, args...; kwargs...) =
    Delta(mutate(rng, state, genotype.change, args...; kwargs...))
