export find, getonly
import Base: map

"""
    noop([x...])

Returns the arguments passed

    noop(::AbstractState, x)

Returns x. Used to perform no operation for retrievers, operators, and updaters.
"""
noop() = nothing
noop(x...) = x
noop(::AbstractState, x) = x

"""
    find(attr::Symbol, match::Any, v::Vector)

Find the first element in v where el.attr == match. Used for finding counters of a specific type.
"""
function find(attr::Symbol, match::Any, v::Vector) 
    for el in v
        if getfield(el, attr) == match
            return el
        end
    end
    @assert false "Failed to retrieve an element from $(typeof(v)) where el.$(attr) == $match"
end

"""
    getonly(f, itr::Union{Vector, Tuple})

Returns the only element in itr that satisfies the predicate f. If there are no elements or more than one element, an error is thrown.
"""
function getonly(f, itr::Union{Vector, Tuple})
    found = filter(f, itr)
    @assert length(found) == 1 "found $(length(found)) items in getonly()"
    found[1]
end

"""
    get_counter(type::Type, state::AbstractState)

Returns the counter of the given type in the state.
"""
get_counter(type::Type, state::AbstractState) = find(:type, type, state.counters)
function get_creators(type::Type, state::AbstractState)
    creators = [c for c in state.creators if c.type <: type]
    @assert !isempty(creators) "Failed to retrieve any creators of type $type"
    creators
end


global env_lock = ReentrantLock()

function get_env_lock()
    if !isdefined(Jevo, :env_lock)
        Jevo.env_lock = ReentrantLock()
    end
    Jevo.env_lock
end

# Returns a mapping function that applies the given op/fn to each element
Base.map(operation::Union{Function, <:AbstractOperator}) =
    (state::AbstractState, objs::Vector) -> [operation(state, obj) for obj in objs]
