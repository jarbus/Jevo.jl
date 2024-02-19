export PopulationRetriever, PopulationCreatorRetriever
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
    (flatten || population.id == id) ? [population] : nothing
end

function get_populations(id::String, populations::Vector{<:AbstractPopulation}; flatten::Bool=false)
    [get_populations(id, p, flatten=flatten) for p in populations] |> filter(!isnothing) |> Iterators.flatten |> collect
end
function get_populations(id::String, composite::CompositePopulation; flatten::Bool=false)
    flatten && composite.id == id && error("Cannot flatten composite population $(composite.id)")
    [get_populations(id, p, flatten=flatten || composite.id == id)
        for p in composite.populations] |> filter(!isnothing) |> Iterators.flatten |> collect
end

"""Traverses the population tree and returns references to all vectors of individuals to an arbitrary depth. If ids is not empty, it only returns individuals from the specified populations"""
function (r::PopulationRetriever)(state::AbstractState)
    if !isempty(r.ids)
        pops = [get_populations(id, state.populations) for id in r.ids]
        @assert !isempty(pops) "Failed to retrieve any populations with ids $(r.ids)"
        @assert all(!isnothing, pops) "Failed to retrieve a population with id $(r.ids)"
    else
        pops = get_populations("", state.populations, flatten=true)
        @assert !isempty(pops) "Failed to retrieve any populations"
    end
    pops
end

"""Retreives all creators of type AbstractPopulation from state.creators"""
struct PopulationCreatorRetriever <: AbstractRetriever end
# (::PopulationCreatorRetriever)(state::AbstractState) = filter(c -> c.type isa AbstractPopulation, state.creators) |> collect
(::PopulationCreatorRetriever)(state::AbstractState) = filter(c -> c.type <: AbstractPopulation, state.creators) |> collect
