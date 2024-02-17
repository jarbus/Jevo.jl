add_matches!(state::AbstractState, matches::Vector{Match}) = append!(state.matches, matches)
struct PopulationAdder <: AbstractUpdater end
(::PopulationAdder)(state::AbstractState, pops::Vector{<:AbstractPopulation}) = append!(state.populations, pops)
