export Population, CompositePopulation
mutable struct CompositePopulation <: AbstractPopulation
    id::String
    populations::Vector{AbstractPopulation}
    data::Vector{AbstractData}
end
# Combine populations into composite
CompositePopulation(id::String, populations::Vector{<:AbstractPopulation}) =
    CompositePopulation(id, populations, Vector{AbstractData}())
# Create composite population using genotype creator
CompositePopulation(id::String,
                    pops::Vector{Tuple{String, Int, Creator}},
                    counters::Vector{<:AbstractCounter}) =
CompositePopulation(id, [Population(id, n, gc, counters) for (id, n, gc) in pops])

mutable struct Population <: AbstractPopulation
    id::String
    population::Vector{Union{AbstractIndividual}}
    data::Vector{AbstractData}
end

# Create pop with predefined inds and no data
Population(id::String, population::Vector{<:AbstractIndividual}) =
    Population(id, population, AbstractData[])

# Create n inds using genotype creator, updates counters
Population(id::String, n::Int, genotype_creator::Creator, counters::Vector{<:AbstractCounter}) =
    Population(id, [Individual(counters, genotype_creator) for i in 1:n])
