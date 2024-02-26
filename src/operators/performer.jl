export Performer

Base.@kwdef struct Performer <: AbstractPerformer
    condition::Function = always
    retriever::Union{AbstractRetriever,Function} = (state::AbstractState) -> state.matches 
    operator::Function = noop
    updater::AbstractUpdater = ComputeInteractions()
    data::Vector{<:AbstractData} = AbstractData[]
    time::Bool = false
end
