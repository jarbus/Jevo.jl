export Individual, develop

"""
    mutable struct Individual{G,D,I} <: AbstractIndividual where 
        {G <: AbstractGenotype,
         D <: AbstractCreator,
         I <: AbstractInteraction}
        id::Int
        generation::Int
        parents::Vector{Int}
        genotype::G
        developer::D
        records::Vector{AbstractRecord}
        interactions::Vector{I}
        data::Vector
    end

An individual is a single entity in the population. Each individual should have a unique id generated from an `AbstractIndividual` [Counter](@ref). An individual's `developer` is a creator that turns a genotype into a phenotype. 
"""
mutable struct Individual{G,D, I} <: AbstractIndividual where 
    {G <: AbstractGenotype,
     D <: AbstractCreator,
     I <: AbstractInteraction}
    id::Int
    generation::Int
    parents::Vector{Int}
    genotype::G
    developer::D
    records::Vector{AbstractRecord}
    interactions::Vector{I}
    data::Vector
end

function Individual(
    id::Int, 
    generation::Int, 
    parents::Vector{Int}, 
    genotype::AbstractGenotype,
    developer::AbstractCreator;
    records::Vector{AbstractRecord} = AbstractRecord[],
    interactions::Vector{<:AbstractInteraction} = AbstractInteraction[],
    data::Vector = []
)
    Individual(id, generation, parents, genotype, developer, records, interactions, data)
end



"""
    Individual(counters::Vector{<:AbstractCounter},
               genotype_creator::Creator,
               developer::AbstractCreator)
Create new individual with no parents
"""
function Individual(counters::Vector{<:AbstractCounter},
           genotype_creator::Creator,
           developer::AbstractCreator
    )
    id = find(:type, AbstractIndividual, counters) |> inc!
    Individual(id, 0, Int[], genotype_creator(), developer)
end

# we allow dispatching on the parameterized individual type, this is the general fallback
develop(creator::Creator, ind::Individual) = develop(creator, ind.genotype)
develop(ind::I) where {I <: AbstractIndividual} = develop(ind.developer, ind)
new_id_and_gen(state::AbstractState) = new_id_and_gen(state.counters)
function new_id_and_gen(counters::Vector{<:AbstractCounter})
    id = find(:type, AbstractIndividual, counters) |> inc!
    gen = find(:type, AbstractGeneration, counters) |> value
    id, gen
end

"""
    clone(state::AbstractState, parent::AbstractIndividual)

Create a child with the same genotype as the parent, but with the id, generation, and parent changed.
"""
function clone(state::AbstractState, parent::AbstractIndividual)
    new_id, new_gen = new_id_and_gen(state)
    Individual(new_id, new_gen, [parent.id], deepcopy(parent.genotype),
               parent.developer)
end

function Base.show(io::IO, ind::Individual)
    print(io, "Individual(id=$(ind.id), gen=$(ind.generation), geno=$(ind.genotype))")
    if !isempty(ind.records)
        print(io, "\n  records: ")
        for record in ind.records
            print(io, "\n    ", record)
        end
    end
    if !isempty(ind.interactions)
        print(io, "\n  interactions: ")
        for (k, v) in ind.interactions
            print(io, "\n    $k: $v")
        end
    end
    if !isempty(ind.data)
        print(io, "\n  data: ")
        for d in ind.data
            print(io, "\n    $d")
        end
    end
end

function serialize_phenotype(path::String, ind::Individual)
    phenotype = develop(ind)
    filepath = joinpath(path, "$(ind.id).jls")
    serialize(filepath, phenotype)
end
