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

Base.@kwdef struct ComputeInteractions <: AbstractUpdater end
function (updater::ComputeInteractions)(::AbstractState, matches::Vector{<:AbstractMatch})
    @assert !isempty(matches) "No matches to compute interactions for"
    for m in matches 
        scores = play(m)
        @assert length(scores) == length(m.individuals) "Number of scores must match number of individuals"
        for i in eachindex(m.individuals)
            ind_id = m.individuals[i].id
            score = scores[i]
            opponent_ids = @inline get_opponent_ids(m, ind_id)
            interaction = Interaction(m.id, ind_id, opponent_ids, score)
            push!(m.individuals[i].interactions, interaction)
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
