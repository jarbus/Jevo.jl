struct Match <: AbstractMatch
    ids::Vector{Int}
    environment_creator::AbstractCreator
    genotypes::Vector{<:AbstractGenotype}
    developers::Vector{<:AbstractCreator}
end
