export Individual, develop

mutable struct Individual{I} <: AbstractIndividual where I <: AbstractInteraction
    id::Int
    generation::Int
    parents::Vector{Int}
    genotype::AbstractGenotype
    developer::AbstractCreator
    records::Vector{AbstractRecord}
    interactions::Vector{I}
    data::Vector{AbstractData}
end

function Individual(
    id::Int, 
    generation::Int, 
    parents::Vector{Int}, 
    genotype::AbstractGenotype,
    developer::AbstractCreator;
    records::Vector{AbstractRecord} = AbstractRecord[],
    interactions::Vector{<:AbstractInteraction} = Interaction[],
    data::Vector{AbstractData} = AbstractData[]
)
    Individual(id, generation, parents, genotype, developer, records, interactions, data)
end



"Create new individual with no parents"
function Individual(counters::Vector{<:AbstractCounter},
           genotype_creator::Creator,
           developer::AbstractCreator
    )
    id = find(:type, AbstractIndividual, counters) |> inc!
    Individual(id, 0, Int[], genotype_creator(), developer)
end

develop(ind::AbstractIndividual) = develop(ind.developer, ind.genotype)
new_id_and_gen(state::AbstractState) = new_id_and_gen(state.counters)
function new_id_and_gen(counters::Vector{<:AbstractCounter})
    id = find(:type, AbstractIndividual, counters) |> inc!
    gen = find(:type, AbstractGeneration, counters) |> value
    id, gen
end

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
