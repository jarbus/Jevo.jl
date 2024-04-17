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
