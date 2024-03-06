export find
import Base: map

noop() = nothing
noop(x...) = x
noop(::AbstractState, x) = x

function find(attr::Symbol, match::Any, v::Vector) 
    for el in v
        if getfield(el, attr) == match
            return el
        end
    end
    @assert false "Failed to retrieve an element from $(typeof(v)) where el.$(attr) == $match"
end

get_counter(type::Type, state::AbstractState) = find(:type, type, state.counters)
function get_creators(type::Type, state::AbstractState)
    creators = [c for c in state.creators if c.type <: type]
    @assert !isempty(creators) "Failed to retrieve any creators of type $type"
    creators
end

function Base.map(operation::Union{Function, <:AbstractOperator})
    return function (state::AbstractState, objs::Vector)
        return [operation(state, obj) for obj in objs]
    end
end

# We need to overwrite this Flux method to generate Float32 weights and maintain compatibility with the (rng, type, dims...) signature
function kaiming_normal(rng::AbstractRNG,::Type, dims::Integer...; gain::Real = √2f0)
  std = Float32(gain / sqrt(first(Flux.nfan(dims...)))) # fan_in
  return randn(rng, Float32, dims...) .* std
end

function apply_kaiming_normal_noise!(rng::AbstractRNG, ::Type, arr::Array{Float32}, mr::Float32; gain::Real = √2f0)
    dims = size(arr)
    std = Float32(gain / sqrt(first(Flux.nfan(dims...))))
    scalar = std * mr
    @fastmath @inbounds @simd for i in 1:length(arr)
        arr[i] += randn(rng, Float32) * scalar
    end
end
