export PopulationAdder, PopulationUpdater, add_matches!, find_population

add_matches!(state::AbstractState, matches::Vector{Match}) = append!(state.matches, matches)

struct PopulationAdder <: AbstractUpdater end
(::PopulationAdder)(state::AbstractState, pops::Vector{<:AbstractPopulation}) =
    append!(state.populations, pops)
    
Base.@kwdef struct PopulationUpdater <: AbstractUpdater
    ids::Vector{String} = String[] # ids of populations to update
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
        for (score, ind) in zip(scores, m.individuals)
            interaction = Interaction(m.id, ind.id, 
                                    [i.id for i in m.individuals if i !== ind], score)
            push!(ind.interactions, interaction)
        end
    end
end
