"""
    struct Match{I,E} <: AbstractMatch where {I <: AbstractIndividual, E <: AbstractEnvironment}
        id::Int
        individuals::Vector{I}
        environment_creator::Creator{E}
    end
"""
struct Match{I,E} <: AbstractMatch where {I <: AbstractIndividual, E <: AbstractEnvironment}
    id::Int
    individuals::Vector{I}
    environment_creator::Creator{E}
end
