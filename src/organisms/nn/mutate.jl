export ComputeMaxMrPerLayerFromGenePool, NNGenePoolMutator, ClearCurrentGenWeights, NNGenePoolReseedMutator

@define_op "ClearCurrentGenWeights" "AbstractMutator"
@define_op "ComputeMaxMrPerLayerFromGenePool"
@define_op "NNGenePoolMutator" "AbstractMutator"
@define_op "NNReseedMutator" "AbstractMutator"


function compute_init(layers)
    n_factors = 0
    for layer in layers
        if layer isa FactorWeight
            n_factors += 1
        elseif layer isa LayerNorm
            strings = [string(typeof(layer)) for layer in layers]
            @error "LayerNorm not supported $strings"
        end
    end
    if n_factors == 0
        return apply_kaiming_normal_noise!
    elseif n_factors == 1
        return apply_kaiming_normal_noise_factored!
    else
        @error "$n_factors factors not supported"
    end
end
function what_layers_should_we_mutate(rng::AbstractRNG, genotype::Network; n::Int, no_layer_norm::Bool)
    n_weights = length(get_weights(genotype, no_layer_norm=no_layer_norm))
    n == -1 && return ones(Bool, n_weights)
    should_mutate = zeros(Bool, n_weights)
    should_mutate[1:n] .= true
    shuffle!(rng, should_mutate)
    should_mutate
end

"""
Uses the Mutation API to clear all weights of the current generation's individuals.

Designed to be used with [Deltas](@ref Delta) which have been cloned.
"""
ClearCurrentGenWeights(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, kwargs...) = 
    create_op("ClearCurrentGenWeights", 
              condition=condition,
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)->mutate!(s, p; fn=clear_weights, kwargs...))),
              time=time;)
clear_weights(::AbstractRNG, ::State, ::Population, genotype) =
    copyarchitecture(genotype)

function mutate(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Network; mr::Union{Float32,Tuple{Vararg{Float32}}}, n::Int=-1, no_layer_norm::Bool=true, kwargs...)
    genotype = deepcopy(genotype)
    gene_counter = get_counter(AbstractGene, state)
    # Choose weights to mutate
    should_mutate = what_layers_should_we_mutate(rng, genotype, n=n, no_layer_norm=no_layer_norm)
    # Determine which weights to mutate based off n
    map!(genotype, weights_only=true) do layers
        weight = layers[end]               
        is_layer_norm(layers) && return     # Skip if we're a layer norm
        !popfirst!(should_mutate) && return # Skip if we don't want to mutate this weight
        init = compute_init(layers)
        mrf0 = mr isa Float32 ? mr : rand(rng, mr)
        push!(weight.muts, NetworkGene(rng, gene_counter, mrf0, init))
    end
    @assert isempty(should_mutate) "Should have iterated through all weights, $(length(should_mutate)) left"
    genotype
end

mutate(rng::AbstractRNG, state::State, population::AbstractPopulation, genotype::Delta, args...; kwargs...) =
    Delta(mutate(rng, state, population, genotype.change, args...; kwargs...))

### ComputeMaxMrPerLayerFromGenePool
"""
Consists of a genotype with at most one gene per weight. 
Each gene has the largest MR in the genepool for that weight.
The ids, seeds, and inits are irrelevant for these.
"""
struct MaxMRs
    maxmrs::Network
end
function compute_max_mr_per_layer!(pop::Population; no_layer_norm::Bool)
    gp = getonly(x->x isa GenePool, pop.data)
    length(gp.deltas) == 0 && return
    max_mr_net = deepcopy(gp.deltas[1].change) # we assume all deltas have same arch
    map(w->empty!(w.muts), Jevo.get_weights(max_mr_net, no_layer_norm=no_layer_norm)) # empty weights of max net
    for delta in gp.deltas
        for (w1, w2) in zip(Jevo.get_weights(max_mr_net, no_layer_norm=no_layer_norm),
                            Jevo.get_weights(delta.change, no_layer_norm=no_layer_norm))
            isempty(w1.muts) && !isempty(w2.muts) && push!(w1.muts, w2.muts[1])
            for mut in w2.muts
                mut.mr > w1.muts[1].mr && (w1.muts[1] = mut)
            end
        end
    end
    lengths = map(w->length(w.muts), Jevo.get_weights(max_mr_net, no_layer_norm=no_layer_norm))
    @assert all(lengths .<= 1) && any(lengths .>= 1) # assert that we have at least one mutation
    filter!(x->!isa(x,MaxMRs), pop.data) # clear prior maxmr
    push!(pop.data, MaxMRs(max_mr_net))
end
ComputeMaxMrPerLayerFromGenePool(ids::Vector{String}=String[];after_gen::Int, no_layer_norm::Bool=true, kwargs...) =
    create_op("ComputeMaxMrPerLayerFromGenePool",
    condition=s->generation(s) > after_gen,
    retriever=PopulationRetriever(ids),
    updater=map(map((_,p)->compute_max_mr_per_layer!(p, no_layer_norm=no_layer_norm))); kwargs...)


########## Max MR Mutator, aka NNGenePoolMutator ##########
function max_mr_mutate(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Network, args...; mr::Union{Float32,Tuple{Vararg{Float32}}}, n::Int=-1, no_layer_norm::Bool, kwargs...)
    genotype = deepcopy(genotype)
    gene_counter = get_counter(AbstractGene, state)
    max_mrs_network = getonly(d->d isa MaxMRs, pop.data).maxmrs
    max_mrs = [!isempty(w.muts) ? w.muts[1].mr : nothing 
               for w in get_weights(max_mrs_network, no_layer_norm=no_layer_norm)]
    # Choose weights to mutate
    should_mutate = what_layers_should_we_mutate(rng, genotype, n=n, no_layer_norm=no_layer_norm)
    # Determine which weights to mutate based off n
    map!(genotype, weights_only=true) do layers
        weight = layers[end]               
        mrf0 = mr isa Float32 ? mr : rand(rng, mr)
        empty!(weight.muts)                 # First, clear copied mutations, since this is a delta
        is_layer_norm(layers) && return     # Skip if we're a layer norm
        max_mr = popfirst!(max_mrs)
        !popfirst!(should_mutate) && return # Skip if we don't want to mutate this weight
        isnothing(max_mr) && return         # Skip if there isn't a max_mr, which implies evolution
                                            # has not selected for evolving this weight in genepool 
        mrf0 > max_mr && return             # Skip if sampled MR is greater than the max
        init = compute_init(layers)
        push!(weight.muts, NetworkGene(rng, gene_counter, mrf0, init))
    end
    @assert isempty(should_mutate) "Should have iterated through all weights, $(length(should_mutate)) left"
    @assert isempty(max_mrs)
    genotype
end
max_mr_mutate(rng::AbstractRNG, state::State, population::AbstractPopulation, genotype::Delta, args...; kwargs...) =
    Delta(max_mr_mutate(rng, state, population, genotype.change, args...; kwargs...))

NNGenePoolMutator(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, kwargs...) = 
    create_op("NNGenePoolMutator", 
              condition=condition,
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)->mutate!(s, p; fn=max_mr_mutate, kwargs...))),
              time=time;)

# ########## seed re-use, aka NNGenePoolReseedMutator ##########
function mutate_reseed(rng::AbstractRNG, state::State, pop::AbstractPopulation, genotype::Network; prob::AbstractFloat, kwargs...)
    genotype = deepcopy(genotype)
    gene_counter = get_counter(AbstractGene, state)
    gene_pool = getonly(d->d isa GenePool, pop.data)

    geno_ws = get_weights(genotype, no_layer_norm=true)
    gp_ws = get_weights(rand(gene_pool.deltas), no_layer_norm=true)
    # Determine which weights to mutate based off n
    for (geno_w, gp_w) in zip(geno_ws, gp_ws)
        length(gp_w.muts) == 0 && continue  # Skip if we don't have any mutations to reseed
        rand(rng) > prob && continue          # Skip if we don't want to reseed
        gene_id = inc!(gene_counter)
        old_mut = gp_w.muts[end]
        new_mut = NetworkGene(gene_id, old_mut.seed, old_mut.mr, old_mut.init!)
        push!(geno_w.muts, new_mut)
    end
    genotype
end

mutate_reseed(rng::AbstractRNG, state::State, population::AbstractPopulation, genotype::Delta, args...; kwargs...) =
    Delta(mutate_reseed(rng, state, population, genotype.change, args...; kwargs...))


"""
    NNGenePoolReseedMutator(ids::Vector{String}=String[]; prob::Float32=0.05, condition=always, time::Bool=false, kwargs...)

"""
NNGenePoolReseedMutator(ids::Vector{String}=String[]; condition::Function=always, time::Bool=false, kwargs...) = 
    create_op("NNReseedMutator", 
              condition=condition,
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)->mutate!(s, p; fn=mutate_reseed, kwargs...))),
              time=time;)
