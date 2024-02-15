
Base.@kwdef mutable struct Individual <: AbstractIndividual
    id::Int
    generation::Int
    parents::Vector{Int}
    genotype::Vector{Int}
    records::Vector{AbstractRecord} = AbstractRecord[]
    interactions::Dict{Any, <:AbstractInteraction} = Dict()
    data::Vector{AbstractData} = AbstractData[]
end
