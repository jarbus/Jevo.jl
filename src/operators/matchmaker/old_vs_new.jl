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
            outcome_cache = outcome_caches[1]
            println("Using outcome cache: $(outcome_cache)")
        end

        @assert length(inds) >= 1
        for ind_i in inds, ind_j in inds
            ind_i.generation != newest_gen && continue
            # randomly sample new vs new candidates
            if ind_i.generation == ind_j.generation 
                push!(random_sample_candidates, [ind_i, ind_j])
                continue
            end
            if !isnothing(outcome_cache) && 
                ind_i.id ∈ keys(outcome_cache.cache) &&
                ind_j.id ∈ keys(outcome_cache.cache) &&
                ind_j.id ∈ keys(outcome_cache.cache[ind_i.id]) &&
                ind_i.id ∈ keys(outcome_cache.cache[ind_j.id])
                continue
            end
            push!(matches, Match(inc!(match_counter), [ind_i, ind_j], env_creator))
            push!(matches, Match(inc!(match_counter), [ind_j, ind_i], env_creator))
        end
        # add random candidates
        shuffle!(random_sample_candidates)
        random_interactions = Tuple{Int, Int}[]
        for i in 1:min(n_randomly_sampled, length(random_sample_candidates))
            ind_i, ind_j = random_sample_candidates[i]
            push!(random_interactions, (ind_i.id, ind_j.id))
            push!(matches, Match(inc!(match_counter), [ind_i, ind_j], env_creator))
        end
        push!(pop.data, RandomlySampledInteractions(pop.id, random_interactions))

        return matches
    end
    @error "Not implemented"
end
