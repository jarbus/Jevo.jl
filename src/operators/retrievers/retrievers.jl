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

Base.in(population::AbstractPopulation, ids::Vector{String})::Bool =
    isempty(ids) || population.id in ids

function get_populations(population::Population,
        ids::Vector{String};
        flatten::Bool=false)
    # If there was a match higher up (flatten=true)
    # or if this is a match, return the population
    flatten || population ∈ ids ?
        AbstractIndividual.(population.individuals) :
        AbstractIndividual[]
end

function get_populations(composite::CompositePopulation, ids::Vector{String}; flatten::Bool=false)
    flatten && composite.id ∈ ids && error("Cannot flatten composite population")
    # if id is matches, return vec{vec{ind}} of all subpops
    # elseif flattening, return vec{ind} of all inds in all subpops
    # else return vec{subpop} of all subpops that match
    if composite.id ∈ ids
        AbstractIndividual[get_populations(p, ids, flatten=true)
                for p in composite.populations] |> filter(!isempty)
    elseif flatten
        [get_populations(p, ids, flatten=flatten) for p in composite.populations] |> vcat |> filter(!isempty)
    else
        [get_populations(p, ids, flatten=flatten) for p in composite.populations] |> filter(!isempty)
    end
end

"""Traverses the population tree and returns references to all vectors of individuals to an arbitrary depth. If ids is not empty, it only returns individuals from the specified populations"""
(r::PopulationRetriever)(state::AbstractState) = get_populations(state.populations, r.ids)

struct PopulationCreatorRetriever <: AbstractRetriever end
(::PopulationCreatorRetriever)(state::AbstractState) = filter(c -> c.type isa AbstractPopulation, state.creators) |> collect
