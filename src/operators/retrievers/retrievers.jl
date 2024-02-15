"""Retrieves Vector{Vector{AbstractIndividual}} from state"""
Base.@kwdef struct PopulationRetreiver <: AbstractRetriever
    ids::Vector{String} = String[] # ids of populations to return
end

Base.in(population::AbstractPopulation, ids::Vector{String})::Bool =
    isempty(ids) || population.id in ids

get_populations(population::Population, ids::Vector{String}) =
    population ∈ ids ? AbstractPopulation[population] : AbstractPopulation[]

get_populations(populations::CompositePopulation, ids::Vector{String}) =
    vcat([get_populations(p, ids) for p in populations.populations]) |> filter(!isempty)

"""Traverses the population tree and returns references to all vectors of individuals to an arbitrary depth. If ids is not empty, it only returns individuals from the specified populations"""
(r::PopulationRetreiver)(state::AbstractState) = get_populations(state.populations, r.ids)

