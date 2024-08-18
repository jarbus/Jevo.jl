export map!, map
import Base: map, map!
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
    map(f::Function, x::Union{AbstractLayer,AbstractWeights,Delta}; weights_only::Bool=false, postordering=false)

Apply a function to all weights in a neural network. Return a vector of the output of each function application in the order which it was applied. See also: [map!](@ref).
"""
function map(f::Function, x::Union{AbstractLayer,AbstractWeights}; kwargs...)
    ret = []
    _map!(f, ret, Any[x]; kwargs...)
    ret
end
map(f::Function, d::Delta, args...; kwargs...) = map(f, d.change, args...; kwargs...)

"""
    map!(f::Function, x::Union{AbstractWeights,AbstractLayer,Delta}; weights_only::Bool=false, postordering=false)

Apply a function to all weights in a neural network. See also: [map](@ref).
"""
map!(f::Function, x::Union{AbstractLayer,AbstractWeights}; kwargs...) = _map!(f, [], Any[x]; kwargs...)
map!(f::Function, d::Delta, args...; kwargs...) = map!(f, d.change, args...; kwargs...)

# Internal mapping function used by map and map!, collects the output of the function in a vector
# but throws it away when used by map!
function _map!(f::Function, ret::Vector, hierarchy::Vector; weights_only::Bool=false, postordering=false)
    length(hierarchy) == 0 && @error "Empty hierarchy"
    skip(hierarchy) && return
    # preorder traversal
    if !postordering
        if hierarchy[end] isa Weights
            push!(ret, f(hierarchy))
            return
        elseif !weights_only
            push!(ret, f(hierarchy))
        end
    end
    # children traversal
    for field in fieldnames(typeof(hierarchy[end]))
        attr = getfield(hierarchy[end], field)
        if typeof(attr) <: Union{Array,Tuple}
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
    # postorder traversal
    if postordering
        if hierarchy[end] isa Weights
            push!(ret, f(hierarchy))
            return
        elseif !weights_only
            push!(ret, f(hierarchy))
        end
    end
end
