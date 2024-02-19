export PopulationRetriever, PopulationCreatorRetriever
"""Retrieves Vector{Vector{AbstractIndividual}} from state

If ids is empty, return all populations
If a composite population in ids, return all individuals in all subpopulations, else return a vector of all sub-populations that match
If a population in ids, return the population

For example, a two-pop all vs all matchmaker with the following population hierarchy: 

    ecosystem
    ├── composite1
    │   ├── pop1a: ind1a1, ind1a2
    │   └── pop1b: ind1b1, ind1b2
    └── composite2
        ├── pop2a: ind2a1, ind2a2
        └── pop2b: ind2b1, ind2b2

ids = ["ecosystem"] or [] will fetch:
    [[inds1a1, ind1a2, ind1b1, ind1b2],
     [ind2a1, ind2a2, ind2b1, ind2b2]]
    which flattens each subpopulations into a single vector
    and returns a vector of these vectors

ids = ["composite1"] or ["pop1a", "pop1b"] will fetch:
    [[ind1a1, ind1a2], [ind1b1, ind1b2]]

ids = ["pop1a"] will fetch:
    [[ind1a1, ind1a2]]
"""

Base.@kwdef struct PopulationRetriever <: AbstractRetriever
    ids::Vector{String} = String[] # ids of populations to return
end

function get_populations(id::String, population::Population; flatten::Bool=false)
    (flatten || population.id == id) ? population.individuals : AbstractIndividual[]
end

function get_populations(id::String, populations::Vector{<:AbstractPopulation}; flatten::Bool=false)
    [get_populations(id, p, flatten=flatten) for p in populations] |> Iterators.flatten |> collect
end
function get_populations(id::String, composite::CompositePopulation; flatten::Bool=false)
    flatten && composite.id == id && error("Cannot flatten composite population $(composite.id)")
    [get_populations(id, p, flatten=flatten || composite.id == id)
        for p in composite.populations] |> Iterators.flatten |> collect
end

"""Traverses the population tree and returns references to all vectors of individuals to an arbitrary depth. If ids is not empty, it only returns individuals from the specified populations"""
(r::PopulationRetriever)(state::AbstractState) = [get_populations(id, state.populations) for id in r.ids]

"""Retreives all creators of type AbstractPopulation from state.creators"""
struct PopulationCreatorRetriever <: AbstractRetriever end
# (::PopulationCreatorRetriever)(state::AbstractState) = filter(c -> c.type isa AbstractPopulation, state.creators) |> collect
(::PopulationCreatorRetriever)(state::AbstractState) = filter(c -> c.type <: AbstractPopulation, state.creators) |> collect
