"""
    get_weights(network::Network; n::Int=-1) -> Array

Returns vector of all weights in a neural network

# Arguments
- `network::Network`: The neural network from which to retrieve the weights.
- `n::Int`: Optional. The index of the specific layer for which to retrieve the weights.
  Provide `-1` (default) to get the weights of all layers.

# Returns
- If `n` is provided and is not `-1`, returns the weights of the `n`-th layer.
- If `n` is `-1`, returns an array of weights for all layers in the network.
"""
function get_weights(rng::AbstractRNG, network::Network; n::Int=-1)
    # get all weights recursively
    weights = Weights[]
    add_weights!(weights, network)
    weights = unique(weights)
    n == -1 && return weights
    shuffle!(rng, weights)
    weights[1:n]
end
add_weights!(weights::Vector{Weights}, ::Any) = nothing
add_weights!(weights::Vector{Weights}, weight::Weights) = push!(weights, weight)
add_weights!(weights::Vector{Weights}, factor::FactorWeight) = add_weights!(weights, [factor.A, factor.B])
add_weights!(weights::Vector{Weights}, comp::CompositeWeight) = add_weights!(weights, comp.weights)
function add_weights!(weights::Vector{Weights}, node::Union{Network,<:AbstractLayer})
    for field in fieldnames(typeof(node))
        add_weights!(weights, getfield(node, field))
    end
end
function add_weights!(weights::Vector{Weights}, arr::Union{Array,Tuple})
    for element in arr
        add_weights!(weights, element)
    end
end
