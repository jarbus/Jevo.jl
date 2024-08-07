export PopulationRetriever, get_individuals
"""Retrieves Vector{Vector{AbstractIndividual}} from state

For example, a two-pop all vs all matchmaker with the following population hierarchy: 

    ecosystem
    ├── composite1
    │   ├── pop1a: ind1a1, ind1a2
    │   └── pop1b: ind1b1, ind1b2
    └── composite2
        ├── pop2a: ind2a1, ind2a2
        └── pop2b: ind2b1, ind2b2

ids = ["ecosystem"] or [] will fetch:
    [inds1a1, ind1a2, ind1b1, ind1b2, ind2a1, ind2a2, ind2b1, ind2b2]]

ids = ["composite1"] will fetch: [ind1a1, ind1a2, ind1b1, ind1b2]]

ids = ["pop1a"] will fetch: [[ind1a1, ind1a2]]
"""
Base.@kwdef struct PopulationRetriever <: AbstractRetriever
    ids::Vector{String} = String[] # ids of populations to return
end

function get_populations(id::String, population::Population; flatten::Bool=false)
    (flatten || population.id == id) ? [population] : Population[]
end

function get_populations(id::String, populations::Vector{<:AbstractPopulation}; flatten::Bool=false)
    [get_populations(id, p, flatten=flatten) for p in populations] |> 
        Iterators.flatten |> collect
end
function get_populations(id::String, composite::CompositePopulation; flatten::Bool=false)
    flatten && composite.id == id && error("Cannot flatten composite population $(composite.id)")
    pops = [get_populations(id, p, flatten=flatten || composite.id == id)
        for p in composite.populations]
    pops = pops |> Iterators.flatten |> collect
    pops
end

"""Traverses the population tree and returns references to all vectors of individuals to an arbitrary depth. If ids is not empty, it only returns individuals from the specified populations"""
function (r::PopulationRetriever)(state::AbstractState, args...)
    if !isempty(r.ids)
        pops = [get_populations(id, state.populations) for id in r.ids]
        @assert !isempty(pops) "Failed to retrieve any populations with ids $(r.ids)"
        @assert all(!isempty, pops) "Failed to retrieve a population with id $(r.ids)"
    else
        pops = [[p] for p in get_populations("", state.populations, flatten=true)]
        @assert !isempty(pops) "Failed to retrieve any populations"
    end
    pops
end

"""
Retreives all creators of type AbstractPopulation from state.creators
"""
struct PopulationCreatorRetriever <: AbstractRetriever end
(::PopulationCreatorRetriever)(state::AbstractState, args...) = filter(c -> c.type <: AbstractPopulation, state.creators) |> collect

# We use args... instead of ::AbstractOperator so it can be used
# outside of the context of an operator
"""
    get_individuals(state::AbstractState, args...)
"""
get_individuals(state::AbstractState, args...) = get_individuals(state.populations)
get_individuals(pop::Population, args...) = pop.individuals
get_individuals(pop::CompositePopulation, args...) = 
    get_individuals.(pop.populations) |> Iterators.flatten |> collect
get_individuals(pops::Vector{<:AbstractPopulation}, args...) =
    get_individuals.(pops) |> Iterators.flatten |> collect

get_timestamps(state::AbstractState, type::Type) = Timestamp[t for t in state.data if t.type == type]
