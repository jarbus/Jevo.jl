struct Match <: AbstractMatch
    id::Int
    individuals::Vector{<:AbstractIndividual}
    environment_creator::AbstractCreator
end
