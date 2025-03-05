export OldVsNewMatchMaker

@define_op "OldVsNewMatchMaker" "AbstractMatchMaker"
OldVsNewMatchMaker(ids::Vector{String}=String[]; no_cached_matches::Bool=false, env_creator=nothing, kwargs...) =
    create_op("OldVsNewMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=(s,ps)->make_old_v_new_matches(s, ps, no_cached_matches; env_creator=env_creator),
          updater=add_matches!;kwargs...)


function make_old_v_new_matches(state::AbstractState, pops::Vector{Vector{Population}}, no_cached_matches::Bool; env_creator=nothing)
    match_counter = get_counter(AbstractMatch, state)
    if isnothing(env_creator)
        env_creators = get_creators(AbstractEnvironment, state)
        @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
        env_creator = env_creators[1]
    end
    matches = Vector{Match}()
    
    pop_ids = [p.id for subpops in pops for p in subpops]
    use_cache = !no_cached_matches
    outcome_cache = getonly(x->x isa OutcomeCache && x.pop_ids == pop_ids , state.data)
    
    if length(pops) == 1 && length(pops[1]) == 1
        newest_gen = maximum(ind.gen for ind in pops[1][1].individuals)
        for subpopi in pops[1]
            inds = subpopi.individuals
            @assert length(inds) >= 1
            for ind_i in inds, ind_j in inds
                ind_i.gen != newest_gen && continue
                ind_i.gen == ind_j.gen  && continue
                if use_cache && 
                    haskey(outcome_cache.cache, (ind_i.id, ind_j.id)) &&
                    haskey(outcome_cache.cache, (ind_j.id, ind_i.id))
                    continue
                end
                push!(matches, Match(inc!(match_counter), [ind_i, ind_j], env_creator))
            end
        end
        return matches
    end
    @error "Not implemented"
end
