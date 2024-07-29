export ElitistLexicaseSelectorAndReproducer

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
@define_op "ElitistLexicaseSelectorAndReproducer" "AbstractOperator"
ElitistLexicaseSelectorAndReproducer(pop_size::Int, ids::Vector{String}=String[]; ϵ::Bool=false, kwargs...) =
    create_op("ElitistLexicaseSelectorAndReproducer",
                    retriever=PopulationRetriever(ids),
                    updater=map((s,p)->lexicase_select!(s,p,pop_size,ϵ))
                    ;kwargs...)
function lexicase_select!(state::AbstractState, pops::Vector{Population}, pop_size::Int, ϵ::Bool)
    @assert pop_size > 0                           "pop_size must be greater than 0"
    @assert length(pops) == 1               "Lexicase selection can only be applied to a non-compsite Population"
    pop = pops[1]
    outcomes = getonly(x->x isa OutcomeMatrix, pop.data).matrix
    @assert !any(isnan, outcomes) "Lexicase selection failed to find a matchup"
    ϵ = if isnothing(ϵ)
        zeros(size(outcomes,2))
    else  # median absolute deviation statistic
         median(abs.(outcomes .- median(outcomes, dims=1)), dims=1) |> vec
    end

    # now that we have our outcomes, lexicase select
    new_pop = Vector{Individual}(undef, pop_size)
    for i in 1:pop_size
        @inline new_pop[i] = pop.individuals[lexicase_sample(state.rng, outcomes, ϵ)]
    end
    # Go through new_pop, replacing any duplicate individuals with clones
    elites = Set{Int}()
    for (idx, ind) in enumerate(new_pop)
        if ind.id ∈ elites
            new_pop[idx] = clone(state, ind)
        else
            push!(elites, ind.id)
            new_pop[idx] = ind
        end
    end
    @assert length(elites) < pop_size "ElitistLexicaseSelector found $(length(elites)) elites for a pop_size=$(pop_size) . This probably shouldn't happen, and you need to change the algorithm if this is."
    pop.individuals[:] = new_pop[:]
end
