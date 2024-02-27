export TruncationSelector
@define_op "TruncationSelector" "AbstractSelector"
TruncationSelector(k::Int, ids::Vector{String}=String[]; kwargs...) =
    create_op("TruncationSelector",
                    retriever=PopulationRetriever(ids),
                    updater=map((s,p)->truncate!(s,p,k))
                    ;kwargs...)
function truncate!(state::AbstractState, pops::Vector{Population}, k::Int)
    @assert k > 0                           "k must be greater than 0"
    @assert length(pops) == 1               "Truncation selection can only be applied to a single Population"
    @assert length(pops[1].individuals) > k "Population must have more individuals than k= $k to truncate, $(pops[1].id) has $(length(pops[1].individuals)) individuals."
    scores = Vector{Float64}(undef, length(pops[1].individuals))
    for (i, ind) in enumerate(pops[1].individuals)
        @assert length(ind.records) == 1 "Individuals must have exactly one record"
        scores[i] = ind.records[1].fitness
    end
    pops[1].individuals = pops[1].individuals[sortperm(scores, rev=true)[1:k]]

end
