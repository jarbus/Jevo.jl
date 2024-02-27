struct Match <: AbstractMatch
    id::Int
    individuals::Vector{<:AbstractIndividual}
    environment_creator::AbstractCreator
end

function get_opponent_ids(match::Match, ind_id::Int)
    opponent_ids = Vector{Int}(undef, length(match.individuals) - 1)
    opp_idx = 1
    @inbounds for i in eachindex(match.individuals)
        opp_id = match.individuals[i].id
        if opp_id != ind_id
            opponent_ids[opp_idx] = opp_id
            opp_idx += 1
        end
    end
    opponent_ids
end
