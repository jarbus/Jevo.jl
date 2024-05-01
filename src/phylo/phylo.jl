export Delta, InitializePhylogeny, UpdatePhylogeny, DeltaCache, InitializeDeltaCache, UpdateDeltaCache
import Base: ==

struct Delta{G} <: AbstractGenotype where {G <: AbstractGenotype}
    change::G 
end

DeltaCache = Dict{Int, Delta}

Base.:(==)(a::Delta, b::Delta) = a.change == b.change

function develop(creator::Creator, ind::Individual{I, G, D}) where {I, G <: Delta, D}
    genotype = worker_construct_child_genome(ind)
    @assert typeof(genotype) <: AbstractGenotype
    develop(creator, genotype)
end

function get_tree(pop::Population)
    trees = filter(p -> p isa PhylogeneticTree, pop.data)
    @assert length(trees) == 1 "Found $(length(trees)) phylogenetic trees for population $(pop.id)"
    tree = first(trees)
    isnothing(tree) && error("No phylogenetic tree found for population $(pop.id)")
    tree
end

function get_delta_cache(pop::Population)
    caches = filter(p -> p isa DeltaCache, pop.data)
    @assert length(caches) == 1 "Found $(length(caches)) delta caches for population $(pop.id)"
    cache = first(caches)
    isnothing(cache) && error("No delta cache found for population $(pop.id)")
    cache
end

function initialize_phylogeny!(::AbstractState, pop::Population)
    @assert isnothing(findfirst(p -> p isa PhylogeneticTree, pop.data)) "Phylogenetic tree already initialized for population $(pop.id)"
    ind_ids = [ind.id for ind in pop.individuals]
    push!(pop.data, PhylogeneticTree(ind_ids))
end

@define_op "InitializePhylogeny"

"""Initializes a phylogenetic tree for each `Population` in the `State`."""
InitializePhylogeny(ids::Vector{String}=String[];kwargs...) = create_op("InitializePhylogeny",
        condition=first_gen,
        retriever=PopulationRetriever(ids),
        updater=map(map((s,p)->initialize_phylogeny!(s, p; kwargs...))))

function update_phylogeny!(state::AbstractState, pop::Population)
    tree = get_tree(pop)
    isnothing(tree) && error("No phylogenetic tree found for population $(pop.id)")
    gen = generation(state)
    for ind in pop.individuals
        if ind.generation == gen
            @assert length(ind.parents) == 1 "Phylo Individuals must have exactly one parent"
            pid = ind.parents[1]
            add_child!(tree, pid, ind.id)
        end
    end
    pop_ids = Set(ind.id for ind in pop.individuals)
    # remove unreachable individuals
    purge_unreachable_nodes!(tree, pop_ids)
    nothing
end

@define_op "UpdatePhylogeny" # add children and prune

UpdatePhylogeny(ids::Vector{String}=String[];kwargs...) = create_op("UpdatePhylogeny",
    retriever=PopulationRetriever(ids),
    updater=map(map((s,p)->update_phylogeny!(s,p)));kwargs...)


@define_op "InitializeDeltaCache"

InitializeDeltaCache(ids::Vector{String}=String[];kwargs...) = create_op("InitializeDeltaCache",
    condition=first_gen,
    retriever=PopulationRetriever(ids),
    updater=map(map((s,p)->(push!(p.data, DeltaCache()); update_delta_cache!(s, p)))); kwargs...)

@define_op "UpdateDeltaCache"

function update_delta_cache!(state::AbstractState, pop::Population)
    tree = get_tree(pop)
    isnothing(tree) && error("No phylogenetic tree found for population $(pop.id)")
    gen = generation(state)
    dc = get_delta_cache(pop)
    # remove deltas for individuals not in phylogeny
    for id in keys(dc)
        if !haskey(tree.tree, id)
            delete!(dc, id)
        end
    end
    # add deltas for individuals in current generation
    for ind in pop.individuals
        if !haskey(dc, ind.id)
            @assert length(ind.parents) <= 1 "Phylo Individuals must have <= 1 parent"
            delta = ind.genotype
            dc[ind.id] = delta
        end
    end
    nothing
end

UpdateDeltaCache(ids::Vector{String}=String[];kwargs...) = create_op("UpdateDeltaCache",
    retriever=PopulationRetriever(ids),
    updater=map(map((s,p)->update_delta_cache!(s,p))); kwargs...)
