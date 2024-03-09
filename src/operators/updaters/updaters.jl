export PopulationAdder, PopulationUpdater, add_matches!, find_population

add_matches!(state::AbstractState, matches::Vector{Match}) = append!(state.matches, matches)


struct PopulationAdder <: AbstractUpdater end
(::PopulationAdder)(state::AbstractState, pops::Vector{<:AbstractPopulation}) =
    append!(state.populations, pops)
    
Base.@kwdef struct PopulationUpdater <: AbstractUpdater
    ids::Vector{String} = String[] # ids of populations to update
end

Base.@kwdef struct RecordAdder <: AbstractUpdater
    ids::Vector{String} = String[] # ids of populations to update
end

function (adder::RecordAdder)(state::AbstractState, records::Vector{<:Vector{<:Vector{<:AbstractRecord}}})
    pops = PopulationRetriever(adder.ids)(state)
    for (subpops, subpops_records) in zip(pops, records)
        @assert length(subpops) == length(subpops_records) "Length of subpopulations and records must match, got $(length(subpops)) and $(length(subpops_records))"
        for (pop, pop_records) in zip(subpops, subpops_records)
            @assert length(pop.individuals) == length(pop_records) "Length of individuals and records must match, got $(length(pop.individuals)) and $(length(pop_records))"
            for (ind, ind_record) in zip(pop.individuals, pop_records)
                @assert length(ind.records) == 0 "Individuals must not have records before adding"
                push!(ind.records, ind_record)
            end
        end
    end
end

function (updater::PopulationUpdater)(state::AbstractState, inds::Vector{<:Vector{<:AbstractIndividual}})
    if isempty(updater.ids)
        pops = PopulationRetriever(updater.ids)(state)
        for (pop, pop_inds) in zip(pops, inds)
            append!(pop.individuals, pop_inds)
        end
    else
        for (id, pop_inds) in zip(updater.ids, inds)
            pop = find_population(id, state)
            append!(pop.individuals, pop_inds)
        end
    end
end

function reset_individual!(ind::AbstractIndividual)
    # Don't shrink arrays when emptying them
    n_interactions = length(ind.interactions)
    n_records = length(ind.records)
    empty!(ind.interactions)
    empty!(ind.records)
    sizehint!(ind.interactions, n_interactions)
    sizehint!(ind.records, n_records)
end

############################
# PERFORMANCE CRITICAL START (measured)
"""Get id of other individual in a 2-player match"""
function get_opponent_ids_2player(match::Match, ind_id::Int)
    ind1_id = match.individuals[1]
    if ind1_id == ind_id
        return Int[match.individuals[2].id]
    end
    Int[match.individuals[1].id]
end

function get_opponent_ids(match::Match, ind_id::Int)
    n_inds = length(match.individuals)
    n_inds == 2 && return @inline get_opponent_ids_2player(match, ind_id)
    n_inds == 1 && return Int[]
    @error("Only 1 or 2 player matches are optimized")
    # The below code is general-purpose, but not optimized
    opponent_ids = Array{Int,1}(undef, n_inds - 1)
    opp_idx = 1
    for opp in match.individuals
        opp_id = opp.id
        if opp_id != ind_id
            opponent_ids[opp_idx] = opp_id
            opp_idx += 1
        end
    end
    opponent_ids
end

function add_interaction_2player!(scores::Vector{T}, match::Match) where T <: AbstractFloat
    m_id = match.id
    inds = match.individuals
    ind1 = inds[1]
    ind2 = inds[2]
    ind1_id = ind1.id
    ind2_id = ind2.id
    int1 = Interaction(m_id, ind1_id, Int[ind2_id], scores[1])
    int2 = Interaction(m_id, ind2_id, Int[ind1_id], scores[2])
    push!(ind1.interactions, int1)
    push!(ind2.interactions, int2)
end
function add_interaction_1player!(scores::Vector{T}, match::Match) where T <: AbstractFloat
    ind = match.individuals[1]
    push!(ind.interactions, Interaction(match.id, ind.id, Int[], scores[1]))
end

function add_interactions!(scores::Vector{T}, match::Match) where T <: AbstractFloat
    n_inds = length(match.individuals)
    @assert length(scores) == n_inds "Number of scores must match number of individuals"
    n_inds == 2 && return @inline add_interaction_2player!(scores, match)
    n_inds == 1 && return @inline add_interaction_1player!(scores, match)
    @error("Only 1 or 2 player matches are optimized")
    # # The below code is general-purpose, but not optimized
    # for i in eachindex(match.individuals)
    #     ind = match.individuals[i]
    #     ind_id = ind.id
    #     opponent_ids = @inline get_opponent_ids(match, ind_id)
    #     push!(ind.interactions, Interaction(match.id, ind_id, opponent_ids, scores[i]))
    # end
    nothing
end

function compute_interaction!(m::Match)
    scores = play(m)
    @inbounds add_interactions!(scores, m)
end
# PERFORMANCE CRITICAL END (measured)
############################
struct ComputeInteractions! <: AbstractUpdater end
function (::ComputeInteractions!)(::AbstractState, matches::Vector{M}) where M <: AbstractMatch
    n_matches = length(matches)
    outs = map(compute_interaction!,matches)
    empty!(matches)
    sizehint!(matches, n_matches)
    outs
end
