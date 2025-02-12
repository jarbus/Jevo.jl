export AllVsAllMatchMaker

"""
    AllVsAllMatchMaker(ids::Vector{String}=String[];kwargs...)

Creates an [Operator](@ref) that creates all vs all matches between individuals in populations with ids in `ids`.
"""
@define_op "AllVsAllMatchMaker" "AbstractMatchMaker"
AllVsAllMatchMaker(ids::Vector{String}=String[];kwargs...) =
    create_op("AllVsAllMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=make_all_v_all_matches,
          updater=add_matches!;kwargs...)


"""
    make_all_v_all_matches(state::AbstractState, pops::Vector{Vector{Population}})

Returns a vector of [Matches](@ref Match) between all pairs of individuals in the populations.

If there is only one population with one subpopulation, it returns a vector of matches between all pairs of individuals in that subpopulation.
"""
function make_all_v_all_matches(state::AbstractState, pops::Vector{Vector{Population}})
    match_counter = get_counter(AbstractMatch, state)
    env_creators = get_creators(AbstractEnvironment, state)
    @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
    env_creator = env_creators[1]
    matches = Vector{Match}()
    
    # if there is only one population with one subpopulation, return all vs all matches between individuals in that subpopulation
    if length(pops) == 1 && length(pops[1]) == 1
        for subpopi in pops[1]
            inds = subpopi.individuals
            @assert length(inds) > 1
            for ind_i in inds, ind_j in inds
                push!(matches, Match(inc!(match_counter), [ind_i, ind_j], env_creator))
            end
        end
        n_inds = length(pops[1][1].individuals)
        @assert length(matches) == n_inds^2
        return matches
    end

    for i in 1:length(pops), j in i+1:length(pops) # for each pair of populations
        for subpopi in pops[i], subpopj in pops[j] # for each pair of subpopulations
            for indi in subpopi.individuals, indj in subpopj.individuals # for each pair of individuals
                push!(matches, Match(inc!(match_counter), [indi, indj], env_creator))
            end
        end
    end
    @assert length(matches) > 0
    matches
end
