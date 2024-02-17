struct All_vs_All <: AbstractMatchMaker
    condition::Function
    retriever::AbstractRetriever
    operator::Function
    updater::AbstractUpdater
    rng::Union{AbstractRNG, Nothing}
    data::Vector{AbstractData}
end

function All_vs_All()
    condition = (::AbstractState) -> true
    retriever = (::AbstractState) -> state.populations
    operator = (pop::AbstractPopulation) -> pop
    updater = add_matches!
    rng = StableRNG(1234)
    All_vs_All(condition, retriever, operator, updater, rng, AbstractData[])
end
