############################
# PERFORMANCE CRITICAL START (measured)
function compute_interactions!(matches::Vector{<:AbstractMatch})
    interactions_vec_vec = pmap(play, matches)  # each match returns multiple interactions
    for (match, interactions_vec) in zip(matches, interactions_vec_vec), 
        ind in unique(match.individuals),
            int in interactions_vec
                ind.id == int.individual_id && push!(ind.interactions, int)
    end
end
# PERFORMANCE CRITICAL END (measured)
############################
"""
"""
struct ComputeInteractions! <: AbstractUpdater end
function (updater::ComputeInteractions!)(::AbstractState, matches::Vector{M}) where M <: AbstractMatch
    n_matches = length(matches)
    compute_interactions!(matches)
    empty!(matches)
    sizehint!(matches, n_matches)
    nothing
end
