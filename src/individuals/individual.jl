export Individual
mutable struct Individual <: AbstractIndividual
    id::Int
    generation::Int
    parents::Vector{Int}
    genotype::AbstractGenotype
    records::Vector{AbstractRecord}
    interactions::Dict{Any, <:AbstractInteraction}
    data::Vector{AbstractData}
end

function Individual(
    id::Int, 
    generation::Int, 
    parents::Vector{Int}, 
    genotype::AbstractGenotype;
    records::Vector{AbstractRecord} = AbstractRecord[],
    interactions::Dict{Any, <:AbstractInteraction} = Dict{Any, AbstractInteraction}(),
    data::Vector{AbstractData} = AbstractData[]
)
    Individual(id, generation, parents, genotype, records, interactions, data)
end
function new_id_and_gen(counters::Vector{<:AbstractCounter})
    id = find(:type, AbstractIndividual, counters) |> inc!
    gen = find(:type, AbstractGeneration, counters) |> value
    id, gen
end

"Create new individual with no parents"
function Individual(counters::Vector{<:AbstractCounter},
           genotype_creator::Creator)
    id, generation = new_id_and_gen(counters)
    Individual(id, generation, Int[], genotype_creator())
end
