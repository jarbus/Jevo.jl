export RandomCohortMatchMaker

@define_op "RandomCohortMatchMaker" "AbstractMatchMaker"
RandomCohortMatchMaker(cohort_size::Int, ids::Vector{String}=String[]; no_cached_matches::Bool=false, n_randomly_sampled::Int=10, env_creator=nothing, kwargs...) =
    create_op("RandomCohortMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=(s,ps)->make_random_cohort_matches(s, ps, cohort_size, no_cached_matches, n_randomly_sampled; env_creator=env_creator),
          updater=add_matches!;kwargs...)


function make_random_cohort_matches(state::AbstractState, pops::Vector{Vector{Population}}, cohort_size::Int, no_cached_matches::Bool, n_randomly_sampled::Int; env_creator=nothing)
    match_counter = get_counter(AbstractMatch, state)
    if isnothing(env_creator)
        env_creators = get_creators(AbstractEnvironment, state)
        @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
        env_creator = env_creators[1]
    end
    matches = Vector{Match}()

    random_sample_candidates = []
    
    if length(pops) == 1 && length(pops[1]) == 1

        pop = pops[1][1]
        inds = pop.individuals
        @assert length(inds) >= 1
        @assert cohort_size <= length(inds) "Cohort size $(cohort_size) is larger than the number of individuals $(length(inds))."
        @assert length(inds) % cohort_size == 0 "Cohort size $(cohort_size) is not a divisor of the number of individuals $(length(inds))."

        outcome_cache = nothing
        use_cached_matches = !no_cached_matches
        if use_cached_matches 
            outcome_caches = filter(x->x isa OutcomeCache && x.pop_ids == [pop.id, pop.id] , state.data)
            @assert length(outcome_caches) == 1 "found $(length(outcome_caches)) output caches."
            outcome_cache = outcome_caches[1].cache
        end

        # split into cohorts
        cohort_indices = shuffle(state.rng, 1:length(inds))
        cohorts = [cohort_indices[i:i+cohort_size-1] for i in 1:cohort_size:length(cohort_indices)]

        # all vs all within cohort
        for cohort in cohorts
            for ind_i_idx in cohort, ind_j_idx in cohort
                ind_i, ind_j = inds[ind_i_idx], inds[ind_j_idx]
                # check if we have already sampled this interaction
                if !isnothing(outcome_cache) && 
                    ind_i.id ∈ keys(outcome_cache) &&
                    ind_j.id ∈ keys(outcome_cache[ind_i.id]) &&
                    ind_i.id ∈ keys(outcome_cache[ind_j.id])
                    continue
                end
                push!(matches, Match(inc!(match_counter), [ind_i, ind_j], env_creator))
            end
        end

        # add random candidates between cohorts
        for i in eachindex(cohorts)
            for j in i+1:length(cohorts)
                for ind_i_idx in cohorts[i], ind_j_idx in cohorts[j]
                    ind_i, ind_j = inds[ind_i_idx], inds[ind_j_idx]
                    push!(random_sample_candidates, (ind_i, ind_j))
                end
            end
        end
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
