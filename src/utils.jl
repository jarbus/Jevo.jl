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

# We need to overwrite this Flux method to generate Float16 weights and maintain compatibility with the (rng, type, dims...) signature
function kaiming_normal(rng::AbstractRNG,::Type, dims::Integer...; gain::Real = √2f0)
  std = Float16(gain / sqrt(first(Flux.nfan(dims...)))) # fan_in
  return randn(rng, Float16, dims...) .* std
end
