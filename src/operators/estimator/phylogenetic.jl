export PhylogeneticEstimator
using DataStructures


struct DistanceStatistics <: AbstractMetric end
struct ErrorStatistics <: AbstractMetric end
struct DistanceErrorCorrelation <: AbstractMetric end

struct NumEstimated <: AbstractMetric end
struct NumEvaluated <: AbstractMetric end
struct NumCached <: AbstractMetric end
struct NumSamples <: AbstractMetric end

@define_op "PhylogeneticEstimator" "AbstractOperator" 
PhylogeneticEstimator(speciesa_id::String, speciesb_id::String, k::Int, max_dist::Int, kwargs...) =
    create_op("PhylogeneticEstimator",
              retriever=PopulationRetriever([speciesa_id, speciesb_id]),
              operator=map((s,ps)->estimate!(s,ps, k=k, max_dist=max_dist)),)

struct OutcomeCache  # LRU
    pop_ids::Vector{String}
    cache::LRU{Int, Dict{Int, Float64}}
end

struct RandomlySampledInteractions
    other_pop_id::String
    interactions::Vector{Tuple{Int, Int}}
end

struct QueueElement
    indA::PhylogeneticNode
    indB::PhylogeneticNode
    dist::Int
    caller::Union{QueueElement, Nothing} # Prevent infinite loops by never returning to the caller
    add_B::Bool # Whether to add inds from tree B
end

struct RelatedOutcome
    """A related outcome is an interaction between two individuals, 
    and the distance to a query pair of individuals. Used for computing
    weighted averages of outcomes.
    """
    ida::Int
    idb::Int
    dist::Int
    outcomea::Float64
    outcomeb::Float64
end

struct EstimatedOutcome
    ida::Int
    idb::Int
    distances::Vector{Int} # the average distance of the k nearest interactions
    est_outcomea::Float64
    est_outcomeb::Float64
end

function EstimatedOutcome(ida::Int,
                          idb::Int,
                          related_outcomes::Vector{RelatedOutcome})
    @assert length(related_outcomes) > 0 "No related outcomes"
    wa, wb = weighted_average_outcome(related_outcomes)
    distances = [r.dist for r in related_outcomes]
    EstimatedOutcome(ida, idb, distances, wa, wb)
end


function estimates_to_outcomes(estimates::Vector{EstimatedOutcome})
    ids = Set(id for e in estimates for id in (e.ida, e.idb))
    individual_outcomes = Dict{Int, Dict{Int, Float64}}(id=>Dict{Int, Float64}() for id in ids)
    for e in estimates
        individual_outcomes[e.ida][e.idb] = e.est_outcomea
        individual_outcomes[e.idb][e.ida] = e.est_outcomeb
    end
    sorted_dict_individual_outcomes = Dict(k=>SortedDict{Int,Float64}(v) for (k,v) in individual_outcomes)
    sorted_dict_individual_outcomes
end


#= function create_individual_outcomes_from_estimates(estimates::Vector{EstimatedOutcome}) =#
#=     ids = Set(id for e in estimates for id in (e.ida, e.idb)) =#
#=     individual_outcomes = Dict{Int, Dict{Int, Float64}}(id=>Dict{Int,Float64}() for id in ids) =#
#=     for e in estimates =#
#=         individual_outcomes[e.ida][e.idb] = e.est_outcomea =#
#=         individual_outcomes[e.idb][e.ida] = e.est_outcomeb =#
#=     end =#
#=     sorted_dict_individual_outcomes = Dict(k=>SortedDict{Int,Float64}(v) for (k,v) in individual_outcomes) =#
#=     sorted_dict_individual_outcomes =#
#= end =#

function measure_estimation_samples(estimates::Vector{EstimatedOutcome},
                         outcomes::Dict{Int, <:AbstractDict{Int, Float64}})
    """Compute metrics of interest for a set of estimates"""
    avg_distances = [mean(e.distances) for e in estimates]
    errorsa = [abs(e.est_outcomea - outcomes[e.ida][e.idb]) for e in estimates]
    errorsb = [abs(e.est_outcomeb - outcomes[e.idb][e.ida]) for e in estimates]
    avg_distances, errorsa, errorsb
end

function two_layer_merge!(d1::AbstractDict{Int, <:AbstractDict}, d2::AbstractDict{Int, <:AbstractDict}; warn::Bool=false)
    """Merge dictionary of dictionaries `d2` into `d1` by merging the inner dictionaries
    if the key is in both dictionaries, and adding the key+dict if it is not in `d1`."""
    for id in keys(d2)
        # If id is in individual_outcomes, merge the two dictionaries using merge!
        if id ∈ keys(d1)
            merge!(d1[id], d2[id])
        else
            warn && @warn "Estimating all outcomes for individual $id"
            d1[id] = d2[id]
        end
    end
end

function weighted_average_outcome(related_outcomes::Vector{RelatedOutcome})
    """Compute the weighted average outcome of a set of related outcomes

    Arguments:
    =========
    related_outcomes: Vector{RelatedOutcome}

    Returns:
    ========
    weighted_average_a, weighted_average_b: Float64, Float64
        The weighted average outcome for individuals a and b
    """
    k = length(related_outcomes)
    if k == 1
        weights = [1.0]
    else
        dists = [related_outcomes[i].dist for i in 1:k]
        inv_dist =  sum(dists) .- dists
        weights = inv_dist ./ sum(inv_dist)
    end
    weighted_average_a, weighted_average_b = 0.0, 0.0
    for i in 1:k
        @inbounds weighted_average_a += related_outcomes[i].outcomea * weights[i]
        @inbounds weighted_average_b += related_outcomes[i].outcomeb * weights[i]
    end
    return weighted_average_a, weighted_average_b
end

function has_outcome(outcomes::AbstractDict{Int, <:AbstractDict{Int, Float64}}, ida::Int, idb::Int)
    """Check if the interaction between `ida` and `idb` is in `outcomes`"""
    return ida ∈ keys(outcomes) && idb ∈ keys(outcomes[ida])
end

function find_k_nearest_interactions(
    ida::Int,
    idb::Int,
    pta::PhylogeneticTree,
    ptb::PhylogeneticTree,
    individual_outcomes::AbstractDict{Int, <:AbstractDict{Int, Float64}},
    k::Int;
    max_dist::Int)
    """Find the k nearest interactions to `ida,idb` in `individual_outcomes`
    by searching over the trees in `pta` and `ptb`.

    This algorithm can also be thought of as a "Dual-Breadth-First-Search", in that
    it traverses two trees, one pair of nodes at a time. To avoid reaching the same
    pair from two different directions, we only search nodes in tree B when the
    current pair was added from a search in tree A.

    Arguments:
    =========
    id1, id2: Int
        The ids of the individuals to find the nearest interactions for
    pta, ptb: PhylogeneticTree
        The phylogenetic trees to search over. id1 must be in pta and id2 must be in ptb
    individual_outcomes: Dict{Int, SortedDict{Int, Float64}}
        A dictionary mapping individual ids to a sorted dictionary of interactions
        and their outcomes. Contains evaluated interactions.
    k: Int
        The number of nearest interactions to find

    Returns:
    ========
    k_nearest_interactions: Vector{RelatedOutcome}
        We return a vector of RelatedOutcome objects instead of a weighted average
        for testing purposes and code-reuse. We can compute different types of weighted
        averages from this vector.
    """
    @assert ida ∈ keys(pta.tree) "id1 $(ida) not in tree A"
    @assert idb ∈ keys(ptb.tree) "id2 $(idb) not in tree B"
    k_nearest_interactions = Vector{RelatedOutcome}()

    # Initialize the search at the interaction
    n1 = pta.tree[ida]
    n2 = ptb.tree[idb]
    # Check if the interaction is already in the dictionary
    if  n2.id in keys(individual_outcomes[n1.id]) && 
        n1.id in keys(individual_outcomes[n2.id])
        error("Interaction $(n1.id),$(n2.id) already in dictionary")
    end
    queue = [QueueElement(n1, n2, 0, nothing, true)]
    iters = 0
    while length(queue) > 0
        el = popfirst!(queue)
        el.dist > max_dist && break
        iters += 1
        ids = (el.indA.id, el.indB.id)
        # If the interaction is in the set of outcomes, add it to the list and break if we have enough
        # @assert ids[1] in keys(individual_outcomes) "id1 $(ids[1]) not in individual_outcomes"
        # @assert ids[2] in keys(individual_outcomes) "id2 $(ids[2]) not in individual_outcomes"

        outcomes1 = get(individual_outcomes[ids[1]], ids[2], nothing)
        if !isnothing(outcomes1)
            outcomes2 = get(individual_outcomes[ids[2]], ids[1], nothing)
            if !isnothing(outcomes2)
                push!(k_nearest_interactions, RelatedOutcome(ids[1], ids[2], el.dist, outcomes1, outcomes2))
                length(k_nearest_interactions) >= k && break
            end
        end

        # Add parent interactions to queue first, as they are more likely to be found:
        # 1. Results from previous generations can be cached
        # 2. For "all vs best" or "all vs parents", we expect parent interactions to 
        #    occur more frequently than child interactions

        # Add Parent of a
        if (isnothing(el.caller) || el.indA.parent != el.caller.indA) && !isnothing(el.indA.parent)
            push!(queue, QueueElement(el.indA.parent, el.indB, el.dist+1, el, false))
        end
        # Add Parent of b if we are supposed to search on the B tree
        if (isnothing(el.caller) || el.indB.parent != el.caller.indB) && !isnothing(el.indB.parent) && el.add_B
            push!(queue, QueueElement(el.indA, el.indB.parent, el.dist+1, el, true))
        end

        # Add children of A
        for neiA in el.indA.children
            !isnothing(el.caller) && neiA == el.caller.indA && continue
            push!(queue, QueueElement(neiA, el.indB, el.dist+1, el, false))
        end
        # Add children of B if we are supposed to search on the B tree
        el.add_B || continue
        for neiB in el.indB.children
            !isnothing(el.caller) &&  neiB == el.caller.indB && continue
            push!(queue, QueueElement(el.indA, neiB, el.dist+1, el, true))
        end
    end
    0 == length(k_nearest_interactions) && error("Found 0 interactions for $(ida),$(idb)")
    return k_nearest_interactions
end

function compute_estimates(
    pairs::Vector{Tuple{Int, Int}},
    treeA::PhylogeneticTree,
    treeB::PhylogeneticTree,
    individual_outcomes::AbstractDict{Int, <:AbstractDict{Int, Float64}};
    k::Int,
    max_dist::Int)
    """For each pair of individuals in `pairs`, find the k nearest interactions
    in `individual_outcomes` and compute the weighted average outcome. 

    Returns:
    ========
    estimates: Vector{EstimatedOutcome}
        A vector of EstimatedOutcome objects for each pair of individuals in `pairs`
    """
    estimates = Vector{EstimatedOutcome}(undef, length(pairs))

    Threads.@threads for i in eachindex(pairs)
        (ida, idb) = pairs[i]
        nearest = find_k_nearest_interactions(ida, idb, treeA, treeB, individual_outcomes, k, max_dist=max_dist)
        estimates[i] = EstimatedOutcome(ida, idb, nearest)
    end
    return estimates
end

function estimate!(state::State, pops::Vector{Population}, k::Int, max_dist::Int)
    # TODO handle case with one pop
    @assert length(pops) == 2 "PhylogeneticEstimator expects two populations"
    popa, popb = pops

    # get outcome cache for this pair of pops
    outcome_cache = nothing
    for d in state.data
        if d isa OutcomeCache && d.pop_ids == [popa.id, popb.id]
            @assert isnothing(outcome_cache)
            outcome_cache = d.cache
        end
    end
    # create if not created
    if isnothing(outcome_cache)
        outcome_cache = LRU{Int, Dict{Int, Float64}}(maxsize=10000)
        push!(state.data, OutcomeCache([popa.id, popb.id], outcome_cache))
    end

    # get randomly sampled interactions for each pop (this should always exist)
    randomly_sampled_interactions_a = getonly(x->x isa RandomlySampledInteractions, popa.data)
    randomly_sampled_interactions_b = getonly(x->x isa RandomlySampledInteractions, popb.data)

    # get trees for each pop
    treea = getonly(x->x isa PhylogeneticTree, popa.data)
    treeb = getonly(x->x isa PhylogeneticTree, popb.data)

    individual_outcomes = Dict{Int, Dict{Int, Float64}}()
    # todo get individual outcomes
    for (i, pop) in enumerate(pops)
        i == 2 && pops[1].id == pops[2].id && continue
        for ind in pop.individuals
            ind_outcomes = individual_outcomes[ind.id] = Dict{Int, Float64}()
            for interaction in ind.interactions
                other_id = interaction.other_ids[1]
                if other_id ∉ keys(ind_outcomes)
                    ind_outcomes[other_id] = 0
                end
                ind_outcomes[other_id] += interaction.score
            end
        end
    end

    # Merge all interactions into outcome cache
    two_layer_merge!(outcome_cache, individual_outcomes)

    # Do we have sampled interactions?
    has_sampled_interactions = any(x->x isa RandomlySampledInteractions && x.other_pop_id == popb.id, popa.data) &&
                               any(x->x isa RandomlySampledInteractions && x.other_pop_id == popa.id, popb.data)
    # If so, remove them from individual_outcomes and outcome cache
    if has_sampled_interactions

        sampled_interactions = [randomly_sampled_interactions_a.interactions; randomly_sampled_interactions_b.interactions] |> unique

        sampled_ids = Set(id for i in sampled_interactions for id in i)

        sampled_individual_outcomes = Dict(id=>SortedDict{Int,Float64}() for id in sampled_ids)
        for (id1, id2) in sampled_interactions
            @assert id1 in keys(individual_outcomes) "id1 $(id1) not in individual_outcomes"
            @assert id2 in keys(individual_outcomes[id1]) "id2 $(id2) not in individual_outcomes[$(id1)]"
            sampled_individual_outcomes[id1][id2] = individual_outcomes[id1][id2]
            sampled_individual_outcomes[id2][id1] = individual_outcomes[id2][id1]
            delete!(individual_outcomes[id1], id2)
            delete!(outcome_cache[id1], id2)
            delete!(individual_outcomes[id2], id1)
            delete!(outcome_cache[id2], id1)
        end
    end


    # Compute all unevaluated interactions between the two species
    unevaluated_interactions = Vector{Tuple{Int64, Int64}}()
    cached_interactions = Vector{Tuple{Int64, Int64}}()

    # Get all cached interactions and all unevaluated outcomes
    for ind_a in popa.individuals, ind_b in popb.individuals
        # We only care about outcomes that haven't been evaluated
        if !has_outcome(individual_outcomes, ind_a.id, ind_b.id)
            if has_outcome(outcome_cache, ind_a.id, ind_b.id)
                push!(cached_interactions, (ind_a.id, ind_b.id))
            else
                push!(unevaluated_interactions, (ind_a.id, ind_b.id))
            end
        end
    end
    
    # Measure the number of cached, estimated, sampled,
    # and evaluated interactions between the two species
    n_cached = length(cached_interactions)
    n_unevaluated = length(unevaluated_interactions)
    n_sampled = has_sampled_interactions ? length(sampled_interactions) : 0
    total_number_of_interactions = length(popa.individuals) * length(popb.individuals)
    num_evaluated = total_number_of_interactions - n_unevaluated - n_cached

    gen = generation(state)
    m = Measurement("PhylogeneticEstimator.$(popa.id*"-"*popb.id).NumCached", n_cached, gen)
    @info m
    @h5 Measurement("PhylogeneticEstimator.$(popa.id*"-"*popb.id).NumCached", n_cached, gen)
    @h5 Measurement("PhylogeneticEstimator.$(popa.id*"-"*popb.id).NumEstimated", n_unevaluated, gen)
    @h5 Measurement("PhylogeneticEstimator.$(popa.id*"-"*popb.id).NumSamples", n_sampled, gen)
    @h5 Measurement("PhylogeneticEstimator.$(popa.id*"-"*popb.id).NumEvaluated", num_evaluated, gen)

    # Skip estimation if there are no unevaluated interactions
    if n_sampled == 0 && n_unevaluated == 0
        return
    end

    # We create a copy of the cached outcomes for use in parallel threads
    # This is because we don't want to lock the cache while we are computing
    # estimates in parallel
    nonlocking_cache = Dict{Int,Dict{Int,Float64}}(k=>copy(v) for (k,v) in outcome_cache)
    # Compute estimates for sampled interactions
    if has_sampled_interactions
        # Estimate sampled interactions
        sample_estimates::Vector{EstimatedOutcome} = compute_estimates(
            sampled_interactions,
            treea,
            treeb,
            nonlocking_cache,
            k=k, max_dist=max_dist)

        sample_estimated_outcomes = estimates_to_outcomes(sample_estimates)
        two_layer_merge!(individual_outcomes, sample_estimated_outcomes, warn=true)

        # Compare sample estimates to actual outcomes
        distances, errs_a, errs_b = measure_estimation_samples(sample_estimates, sampled_individual_outcomes)

        @h5 StatisticalMeasurement("estimation_distances", distances, gen)
        @h5 StatisticalMeasurement("estimation_errors_a", errs_a, gen)
        @h5 StatisticalMeasurement("estimation_errors_b", errs_b, gen)
        @h5 Measurement("estimation_distance_error_correlation_a", cor(distances, errs_a), gen)
        @h5 Measurement("estimation_distance_error_correlation_b", cor(distances, errs_b), gen)
    end

    # Compute estimates for all unevaluated interactions
    estimates = compute_estimates(
                            unevaluated_interactions,
                            treea,
                            treeb,
                            nonlocking_cache,
                            k=k, max_dist=max_dist)
    estimated_individual_outcomes = estimates_to_outcomes(estimates)
    # merge estimated_individual_outcomes into individual_outcomes
    #two_layer_merge!(individual_outcomes, estimated_individual_outcomes, warn=true)

    # Add cached interactions to individual_outcomes
    for (id1, id2) in cached_interactions
        individual_outcomes[id1][id2] = outcome_cache[id1][id2]
        individual_outcomes[id2][id1] = outcome_cache[id2][id1]
    end

    # for each ind in each pop, add the estimated outcomes to the interactions
    for (i, pop) in enumerate(pops)
        # skip duplicate add if we are estimating interactions within the same population
        i == 2 && pops[1].id == pops[2].id && continue
        for ind in pop.individuals, (other_id, outcome) in estimated_individual_outcomes[ind.id]
            push!(ind.interactions, EstimatedInteraction(ind.id, [other_id], outcome))
        end
    end
    # for each ind in each pop confirm that they have 
    # at least one interaction with every member of the other pop
    pop_ids = [Set(ind.id for ind in pop.individuals) for pop in pops]
    for (i, pop) in enumerate(pops)
        i == 2 && pops[1].id == pops[2].id && continue
        for ind in pop.individuals
            found_other_ids = Set(int.other_ids[1] for int in ind.interactions)
            other_pop = pop_ids[3-i]
            missing_other_ids = setdiff(other_pop, found_other_ids)
            @assert isempty(missing_other_ids) "Individual $(ind.id) is missing interactions with $(missing_other_ids)"
        end
    end
end

@define_op "RestoreCachedInteractions" "AbstractOperator" 
RestoreCachedInteractions(ids::Vector{String}=String[], kwargs...) =
    create_op("RestoreCachedInteractions",
              retriever=PopulationRetriever(ids),
              operator=restore_cached_outcomes!,
    )

function restore_cached_outcomes!(state::State, pops::Vector{Vector{Population}})
    length(pops) != 1 && @error "Not implemented"
    pop = pops[1][1]
    outcome_cache = getonly(x->x isa OutcomeCache && x.pop_ids == [pop.id], state.data).cache
    ind_ids = Set(ind.id for ind in pop.individuals)
    for ind in pop.individuals
        other_ids = Set(int.other_ids[1] for int in ind.interactions)
        ind_outcome_cache = outcome_cache[ind.id]
        for other_id in setdiff(ind_ids, other_ids)
            if other_id in keys(ind_outcome_cache)
                push!(ind.interactions, Interaction(ind.id, [other_id], outcome_cache[ind.id][other_id]))
            end
        end
    end
end
