export AllVsAllMatchMaker

"""
    AllVsAllMatchMaker(ids::Vector{String}=String[];kwargs...)

Creates an [Operator](@ref) that creates all vs all matches between individuals in populations with ids in `ids`.
"""
@define_op "AllVsAllMatchMaker" "AbstractMatchMaker"
AllVsAllMatchMaker(ids::Vector{String}=String[]; use_cache=false, env_creator=nothing, kwargs...) =
    create_op("AllVsAllMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=(s,ps)->make_all_v_all_matches(s, ps, env_creator=env_creator, use_cache=use_cache),
          updater=add_matches!;kwargs...)


"""
    make_all_v_all_matches(state::AbstractState, pops::Vector{Vector{Population}})

Returns a vector of [Matches](@ref Match) between all pairs of individuals in the populations.

If there is only one population with one subpopulation, it returns a vector of matches between all pairs of individuals in that subpopulation.
"""
function make_all_v_all_matches(state::AbstractState, pops::Vector{Vector{Population}}; env_creator=nothing, use_cache=false)
    match_counter = get_counter(AbstractMatch, state)
    if isnothing(env_creator)
        env_creators = get_creators(AbstractEnvironment, state)
        @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
        env_creator = env_creators[1]
    end
    matches = Vector{Match}()

    # if there is only one population with one subpopulation, return all vs all matches between individuals in that subpopulation
    if length(pops) == 1 && length(pops[1]) == 1
        pop = pops[1][1]
        pop_ids = [pop.id, pop.id]
        outcome_caches = filter(x->x isa OutcomeCache && x.pop_ids == pop_ids , state.data)
        if length(outcome_caches) == 0
            push!(state.data, OutcomeCache(pop_ids, LRU{Int, Dict{Int, Float64}}(maxsize=100_000)))
        else
            @assert length(outcome_caches) == 1 "There should be exactly one outcome cache for the time being, found $(length(outcome_caches))."
        end
        outcome_cache = !use_cache || isempty(outcome_caches) ? nothing : outcome_caches[1].cache

        inds = pop.individuals
        @assert length(inds) >= 1
        for ind_i in inds, ind_j in inds

            check_if_outcome_in_cache(outcome_cache, ind_i.id, ind_j.id) && continue
            push!(matches, Match(Jevo.inc!(match_counter), [ind_i, ind_j], env_creator))
        end
        @info "Created $(length(matches)) matches"
        return matches
    end

    for i in 1:length(pops), j in i+1:length(pops) # for each pair of populations
        for subpopi in pops[i], subpopj in pops[j] # for each pair of subpopulations
            for indi in subpopi.individuals, indj in subpopj.individuals # for each pair of individuals
                push!(matches, Match(Jevo.inc!(match_counter), [indi, indj], env_creator))
            end
        end
    end
    @assert length(matches) > 0
    matches
end
