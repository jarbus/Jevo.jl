export PopulationAdder!, PopulationUpdater!, add_matches!, find_population

"""
    add_matches!(state::AbstractState, matches::Vector{Match})

Updater which adds matches to the state.
"""
add_matches!(state::AbstractState, matches::Vector{Match}) = append!(state.matches, matches)


"""
    PopulationAdder!()

Creates an [Updater](@ref) that adds populations to the state.
"""
struct PopulationAdder! <: AbstractUpdater end
function (::PopulationAdder!)(state::AbstractState, pops::Vector{<:AbstractPopulation})
    @assert length(state.populations) == 0 "We only expect to add populations to the state once"
    append!(state.populations, pops)
end
    
"""
    PopulationUpdater!(;ids=String[])

Currently unimplemented.
"""
Base.@kwdef struct PopulationUpdater! <: AbstractUpdater
    ids::Vector{String} = String[] # ids of populations to update
end

"""
    ReccordAdder!(;ids=String[])

Creates an [Updater](@ref) that adds records to individuals in populations with ids in `ids`.

    function(adder::ReccordAdder!)(state::AbstractState, records::Vector{<:Vector{<:Vector{<:AbstractRecord}}})
"""
Base.@kwdef struct ReccordAdder! <: AbstractUpdater
    ids::Vector{String} = String[] # ids of populations to update
end

function (adder::ReccordAdder!)(state::AbstractState, records::Vector{<:Vector{<:Vector{<:AbstractRecord}}})
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

function (updater::PopulationUpdater!)(state::AbstractState, inds::Vector{<:Vector{<:AbstractIndividual}})
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


