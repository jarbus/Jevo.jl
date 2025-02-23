export SelfVsSelfMatchMaker

"""
    SelfVsSelfMatchMaker(ids::Vector{String}=String[];kwargs...)

Creates an [Operator](@ref) that creates all vs all matches between individuals in populations with ids in `ids`.
"""
@define_op "SelfVsSelfMatchMaker" "AbstractMatchMaker"
SelfVsSelfMatchMaker(ids::Vector{String}=String[]; env_creator=nothing, kwargs...) =
    create_op("SelfVsSelfMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=(s,ps)->make_self_vs_self_matches(s, ps; env_creator=env_creator),
          updater=add_matches!;kwargs...)


"""
    make_self_vs_self_matches(state::AbstractState, pops::Vector{Vector{Population}})

Returns a vector of [Matches](@ref Match) between all pairs of individuals in the populations.

If there is only one population with one subpopulation, it returns a vector of matches between all pairs of individuals in that subpopulation.
"""
function make_self_vs_self_matches(state::AbstractState, pops::Vector{Vector{Population}}; env_creator=nothing)
    match_counter = get_counter(AbstractMatch, state)
    if isnothing(env_creator)
        env_creators = get_creators(AbstractEnvironment, state)
        @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
        env_creator = env_creators[1]
    end
    matches = Vector{Match}()
    
    # if there is only one population with one subpopulation, return all vs all matches between individuals in that subpopulation
    @assert length(pops) == 1 && length(pops[1]) == 1
    for subpopi in pops[1]
        inds = subpopi.individuals
        @assert length(inds) > 0
        for ind_i in inds
            push!(matches, Match(inc!(match_counter), [ind_i, ind_i], env_creator))
        end
    end
    n_inds = length(pops[1][1].individuals)
    @assert length(matches) == n_inds
    matches
end
