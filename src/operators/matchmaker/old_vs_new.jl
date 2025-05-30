export OldVsNewMatchMaker

@define_op "OldVsNewMatchMaker" "AbstractMatchMaker"
OldVsNewMatchMaker(ids::Vector{String}=String[]; no_cached_matches::Bool=false, n_randomly_sampled::Int=10, env_creator=nothing, kwargs...) =
    create_op("OldVsNewMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=(s,ps)->make_old_vs_new_matches(s, ps, no_cached_matches, n_randomly_sampled; env_creator=env_creator),
          updater=add_matches!;kwargs...)


function make_old_vs_new_matches(state::AbstractState, pops::Vector{Vector{Population}}, no_cached_matches::Bool, n_randomly_sampled::Int; env_creator=nothing)
    match_counter = get_counter(AbstractMatch, state)
    if isnothing(env_creator)
        env_creators = get_creators(AbstractEnvironment, state)
        @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
        env_creator = env_creators[1]
    end
    matches = Vector{Match}()

    random_sample_candidates = []
    
    if length(pops) == 1 && length(pops[1]) == 1

        newest_gen = maximum(ind.generation for ind in pops[1][1].individuals)
        pop = pops[1][1]
        inds = pop.individuals


        outcome_cache = nothing
        use_cached_matches = !no_cached_matches
        if use_cached_matches 
            outcome_caches = filter(x->x isa OutcomeCache && x.pop_ids == [pop.id, pop.id] , state.data)
            @assert length(outcome_caches) == 1 "found $(length(outcome_caches)) output caches."
            outcome_cache = outcome_caches[1].cache
        end

        @assert length(inds) >= 1
        for ind_i in inds, ind_j in inds
            # randomly sample new vs new candidates
            if ind_i.generation == ind_j.generation == newest_gen
                push!(random_sample_candidates, [ind_i, ind_j])
                continue
            end
            check_if_outcome_in_cache(outcome_cache, ind_i.id, ind_j.id) && continue
            push!(matches, Match(inc!(match_counter), [ind_i, ind_j], env_creator))
        end
        # add random candidates
        random_sample_candidates = shuffle(state.rng, unique(random_sample_candidates))
        random_interactions = Tuple{Int, Int}[]
        for i in 1:min(n_randomly_sampled, length(random_sample_candidates))
            ind_i, ind_j = random_sample_candidates[i]
            push!(random_interactions, (ind_i.id, ind_j.id))
            ind_i.id != ind_j.id && push!(random_interactions, (ind_j.id, ind_i.id))
            push!(matches, Match(inc!(match_counter), [ind_i, ind_j], env_creator))
        end

        previous_random_interactions = [d for d in pop.data if d isa RandomlySampledInteractions && d.other_pop_id == pop.id]
        if length(previous_random_interactions) > 1
            @error "Found more than one RandomlySampledInteractions for population $(pop.id)."
        elseif length(previous_random_interactions) == 0
            push!(pop.data, RandomlySampledInteractions(pop.id, random_interactions))
        else
            append!(previous_random_interactions[1].interactions, random_interactions)
        end

        return matches
    end
    @error "Not implemented"
end
