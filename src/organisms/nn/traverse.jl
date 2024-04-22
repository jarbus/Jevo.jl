export map!, map
import Base: map, map!
# """
#     get_weights(network::Network; n::Int=-1) -> Array
#
# Returns vector of all weights in a neural network
#
# # Arguments
# - `network::Network`: The neural network from which to retrieve the weights.
# - `n::Int`: Optional. The index of the specific layer for which to retrieve the weights.
#   Provide `-1` (default) to get the weights of all layers.
#
# # Returns
# - If `n` is provided and is not `-1`, returns the weights of the `n`-th layer.
# - If `n` is `-1`, returns an array of weights for all layers in the network.
# """
# function get_weights(rng::AbstractRNG, network::Network; n::Int=-1)
#     # get all weights recursively
#     weights = Weights[]
#     add_weights!(weights, network)
#     weights = unique(weights)
#     n == -1 && return weights
#     shuffle!(rng, weights)
#     weights[1:n]
# end
# add_weights!(weights::Vector{Weights}, ::Any) = nothing # don't recurse for anything that's not a weight
# add_weights!(weights::Vector{Weights}, weight::Weights) = push!(weights, weight)
# add_weights!(weights::Vector{Weights}, factor::FactorWeight) = add_weights!(weights, [factor.A, factor.B])
# add_weights!(weights::Vector{Weights}, comp::CompositeWeight) = add_weights!(weights, comp.weights)
# add_weights!(::Vector{Weights}, ::LayerNorm) = nothing
# function add_weights!(weights::Vector{Weights}, node::Union{Network,<:AbstractLayer})
#     for field in fieldnames(typeof(node))
#         add_weights!(weights, getfield(node, field))
#     end
# end
# function add_weights!(weights::Vector{Weights}, arr::Union{Array,Tuple})
#     for element in arr
#         add_weights!(weights, element)
#     end
# end
#

"""
    skip iterating over certain hierarchical configurations. We currently use this to
avoid iterating over the embedding weights for both the Embed and EmbedDecoder layers.
"""
function skip(hierarchy::Vector)
    embed_decoder_in_hierarchy = false
    embed_in_hierarchy = false
    for layer in hierarchy
        if layer isa EmbedDecoder
            embed_decoder_in_hierarchy = true
        elseif layer isa Embed
            embed_in_hierarchy = true
        end
    end
    embed_decoder_in_hierarchy && embed_in_hierarchy && return true
end

"""
    map!(f::Function, x::AbstractLayer)

Apply a function to all weights in a neural network.
"""
map!(f::Function, x::AbstractLayer; weights_only::Bool=false) = _map!(f, Any[x], weights_only=weights_only)

function _map!(f::Function, hierarchy::Vector; weights_only::Bool)
    length(hierarchy) == 0 && @error "Empty hierarchy"
    skip(hierarchy) && return
    if hierarchy[end] isa Weights
        f(hierarchy)
        return
    elseif !weights_only
        f(hierarchy)
    end
    for field in fieldnames(typeof(hierarchy[end]))
        attr = getfield(hierarchy[end], field)
        if typeof(attr) <: Union{Vector,Tuple}
            for element in attr
                push!(hierarchy, element)
                _map!(f, hierarchy, weights_only=weights_only)
                pop!(hierarchy)
            end
        else
            push!(hierarchy, attr)
            _map!(f, hierarchy, weights_only=weights_only)
            pop!(hierarchy)
        end
    end
end

function map(f::Function, x::AbstractLayer; weights_only::Bool=false)
    ret = []
    _map!(f, ret, Any[x], weights_only=weights_only)
    ret
end

function _map!(f::Function, ret::Vector, hierarchy::Vector; weights_only::Bool)
    length(hierarchy) == 0 && @error "Empty hierarchy"
    skip(hierarchy) && return
    if hierarchy[end] isa Weights
        push!(ret, f(hierarchy))
        return
    elseif !weights_only
        push!(ret, f(hierarchy))
    end
    for field in fieldnames(typeof(hierarchy[end]))
        attr = getfield(hierarchy[end], field)
        if typeof(attr) <: Union{Vector,Tuple}
            for element in attr
                push!(hierarchy, element)
                _map!(f, ret, hierarchy, weights_only=weights_only)
                pop!(hierarchy)
            end
        else
            push!(hierarchy, attr)
            _map!(f, ret, hierarchy, weights_only=weights_only)
            pop!(hierarchy)
        end
    end
end
map(f::Function, d::Delta, args...; kwargs...) = map(f, d.change, args...; kwargs...)
map!(f::Function, d::Delta, args...; kwargs...) = map!(f, d.change, args...; kwargs...)
