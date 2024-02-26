export TruncationSelector
struct TruncationSelector <: AbstractSelector
    condition::Function
    retriever::AbstractRetriever
    operator::Function
    updater::Function
    time::Bool
end

function truncate!(state::AbstractState, pops::Vector{Population}, k::Int)
    @assert k > 0                           "k must be greater than 0"
    @assert length(pops) == 1               "Truncation selection can only be applied to a single Population"
    @assert length(pops[1].individuals) > k "Population must have more individuals than k= $k to truncate, $(pops[1].id) has $(length(pops[1].individuals)) individuals."
    scores = Vector{Float32}(undef, length(pops[1].individuals))
    for (i, ind) in enumerate(pops[1].individuals)
        @assert length(ind.records) == 1 "Individuals must have exactly one record"
        scores[i] = ind.records[1].fitness
    end
    pops[1].individuals = pops[1].individuals[sortperm(scores, rev=true)[1:k]]

end

function TruncationSelector(k::Int, ids::Vector{String}=String[]; time::Bool=false)
    TruncationSelector(always,
                       PopulationRetriever(ids),
                       noop,
                       map((s,p)->truncate!(s,p,k)),
                       time)
end
