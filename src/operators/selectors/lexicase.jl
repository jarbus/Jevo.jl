export LexicaseSelectorAndReproducer, ClearMissingInteractions, ClearOutcomeMatrix

function fast_max_filter!(source_idxs::Vector{Int},
        n_source_idxs::Int,
        target_idxs::Vector{Int},
        outcomes::Matrix{Float64},
        ϵ::Float64,
        test_idx::Int)
    """Copies source_ids with the max outcome for a given test into the into first slots of target_ids."""
    cur_max = -Inf
    n_target_idxs = 0
    for i in 1:n_source_idxs
        @inbounds idx = source_idxs[i]
        @inbounds outcome = outcomes[idx, test_idx] # This is the lexicase selection bottleneck
        # Reset targets if we find new max
        if outcome > cur_max + ϵ
            cur_max = outcome
            @inbounds target_idxs[1] = idx
            n_target_idxs = 1
        # Add to targets if we find another max
        elseif outcome > cur_max - ϵ
            n_target_idxs += 1
            @inbounds target_idxs[n_target_idxs] = idx
        end
    end
    @assert cur_max != -Inf
    return n_target_idxs
end

function lexicase_sample(
    rng::AbstractRNG,
    outcomes::Matrix{Float64},
    ϵ::Vector{Float64})

    source_idxs = collect(1:size(outcomes, 1))
    target_idxs = Vector{Int}(undef, length(source_idxs))
    n_source_idxs = length(source_idxs)
    test_idxs = Set(1:size(outcomes, 2))

    while n_source_idxs > 1 && length(test_idxs) > 0
        # sample a test case without replacement
        rand_test_idx = pop!(test_idxs, rand(rng, test_idxs))
        # out of the remaining ids, choose the ones that have the best fitness on the test case
        n_source_idxs = @inline fast_max_filter!(source_idxs, n_source_idxs, target_idxs, outcomes, ϵ[rand_test_idx], rand_test_idx)
        # swap source and target ids
        source_idxs, target_idxs = target_idxs, source_idxs
    end
    # There is either one id left, or no test cases left. For both cases we can choose one at random
    @assert all(source_idxs[1:n_source_idxs] .<= size(outcomes, 1))
    rand(rng, source_idxs[1:n_source_idxs])
end

"""
    struct ElitistLexicaseSelectorAndReproducer <: Operator
        pop_size::Int         # number of parents to select
        ϵ::Bool  # whether to perform epsilon-lexicase selection (for continuous domains)
    end

Updates the population with selected individuals using (ϵ)-lexicase selection[1]

[1] A probabilistic and multi-objective analysis of lexicase selection and ε-lexicase selection. La Cava et al (2019)
"""
@define_op "LexicaseSelectorAndReproducer" "AbstractOperator"
LexicaseSelectorAndReproducer(pop_size::Int, ids::Vector{String}=String[]; ϵ::Bool=false, elitism::Bool=false, selection_only::Bool=false, keep_all_parents::Bool=false, h5::Bool=false, kwargs...) =
    create_op("LexicaseSelectorAndReproducer",
                    retriever=PopulationRetriever(ids),
                    updater=map(map((s,p)->lexicase_select!(s,p,pop_size,ϵ, elitism, selection_only, keep_all_parents))),
                    ;kwargs...)
function lexicase_select!(state::AbstractState, pop::Population, pop_size::Int, ϵ::Bool, elitism::Bool, selection_only::Bool, keep_all_parents::Bool)
    @assert !selection_only || elitism "You probably don't want to use selection_only without elitism"
    @assert !(selection_only && keep_all_parents) "You probably don't want to use selection_only and keep_all_parents"
    @assert pop_size > 0                           "pop_size must be greater than 0"
    outcomes = getonly(x->x isa OutcomeMatrix, pop.data).matrix
    @assert !any(isnan, outcomes) "Lexicase selection failed to find a matchup"
    ϵ = if !ϵ 
        zeros(size(outcomes,2)) 
    else
        median(abs.(outcomes .- median(outcomes, dims=1)), dims=1) |> vec
    end

    if keep_all_parents
        new_pop = deepcopy(pop.individuals)
        start_ind = length(new_pop) + 1
    else
        new_pop = Vector{Individual}()
        start_ind = 1
    end

    # now that we have our outcomes, lexicase select
    for _ in start_ind:pop_size
        new_ind =  pop.individuals[lexicase_sample(state.rng, outcomes, ϵ)]
        if !selection_only || new_ind ∉ new_pop
            @inline push!(new_pop, new_ind)
        end
    end
    elites = Set{Int}()
    # Go through new_pop, replacing any duplicate individuals with clones
    for (idx, ind) in enumerate(new_pop)
        if !elitism || ind.id ∈ elites
            new_pop[idx] = clone(state, ind)
        else
            push!(elites, ind.id)
        end
    end

    if elitism
        length(elites) <= 2 && @info "WARNING: Found <= 2 elites: $(elites)"
        @assert length(elites) < pop_size "ElitistLexicaseSelector found $(length(elites)) elites for a pop_size=$(pop_size) . This probably shouldn't happen, and you need to change the algorithm if this is."
        #= @info "selected elites $elites with parents $parents" =#
        @info "selected $(length(elites)) elites"
        h5 && @h5 Measurement("n_elites", length(elites), generation(state))
    end
    pop.individuals = new_pop
end


ClearOutcomeMatrix(ids::Vector{String}=String[]; kwargs...) =
    create_op("LexicaseSelectorAndReproducer",
                    retriever=PopulationRetriever(ids),
                    updater=map(map((s,p)->filter!(x->!isa(x, OutcomeMatrix), p.data))),
                    ;kwargs...)

ClearMissingInteractions(ids::Vector{String}=String[]; kwargs...) =
    create_op("LexicaseSelectorAndReproducer",
                    retriever=PopulationRetriever(ids),
                    updater=map(map((s,p)->clear_missing_interactions!(s,p))),
                    ;kwargs...)

function clear_missing_interactions!(state, pop)
    # confirm all invidivudals have at least one interaction
    @assert all(length(ind.interactions) > 0 for ind in pop.individuals)
    ind_ids = Set(ind.id for ind in pop.individuals)
    # remove any interaction where all ids in interation.other_ids is not in ind_ids
    for ind in pop.individuals 
        filter!(int->all(id->id in ind_ids, int.other_ids), ind.interactions)
    end
    @assert all(length(ind.interactions) > 0 for ind in pop.individuals) "This is only supported for one population right now"
end
