export Population, CompositePopulation
import Base: show
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
    individuals::Vector{AbstractIndividual}
    data::Vector{AbstractData}
end

# Create pop with predefined inds and no data
Population(id::String, individuals::Vector{<:AbstractIndividual}) =
    Population(id, individuals, AbstractData[])

# Create n inds using genotype creator, updates counters
Population(id::String, n::Int, genotype_creator::Creator, counters::Vector{<:AbstractCounter}) =
    Population(id, [Individual(counters, genotype_creator) for i in 1:n])

# composite pop
function Base.show(io::IO, pop::CompositePopulation; depth::Int=0)
    print(io, pop.id, ": ", length(pop.populations), " populations:")
    for subpop in pop.populations
        print(io, "\n", " "^(depth+2))
        Base.show(io, subpop, depth=depth+2)
    end
end

function Base.show(io::IO, pop::Population; depth::Int=0)
    print(io, pop.id, ": ", length(pop.individuals), " individuals:")
    for ind in pop.individuals
        print(io, "\n", " "^(depth+2), ind)
    end
end
