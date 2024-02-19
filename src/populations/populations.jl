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
    individuals::Vector{AbstractIndividual}
    data::Vector{AbstractData}
end

# Create pop with predefined inds and no data
Population(id::String, individuals::Vector{<:AbstractIndividual}) =
    Population(id, individuals, AbstractData[])

# Create n inds using genotype creator, updates counters
Population(id::String, n::Int, genotype_creator::Creator, counters::Vector{<:AbstractCounter}) =
    Population(id, [Individual(counters, genotype_creator) for i in 1:n])

find_population(id::String, population::Population) = 
    population.id == id ? population : nothing

function find_population(id::String, composite::CompositePopulation)
    if composite.id == id
        @error "Cannot retrieve a composite population with id $id"
    end
    for pop in composite.populations
        found_pops = find_population(id, pop)
        if !isnothing(found_pops)
            return found_pops
        end
    end
    nothing
end

function find_population(id::String, state::AbstractState)
    pops =[find_population(id, p) for p in state.populations] |>
        filter(!isnothing)
    @assert length(pops) == 1 "Failed to retrieve a single population with id $id"
    pops[1]
end

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
