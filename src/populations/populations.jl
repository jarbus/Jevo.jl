Base.@kwdef mutable struct CompositePopulation <: AbstractPopulation
    id::String
    populations::Vector{AbstractPopulation}
    data::Vector{AbstractData} = AbstractData[]
end

Base.@kwdef mutable struct Population <: AbstractPopulation
    id::String
    population::Vector{Union{AbstractIndividual}}
    data::Vector{AbstractData} = AbstractData[]
end
