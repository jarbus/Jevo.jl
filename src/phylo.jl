export Delta, InitializePhylogeny, UpdatePhylogeny, PurgePhylogeny, DeltaCache, InitializeDeltaCache, UpdateDeltaCache, UpdateGenePool, GenePool, TrackPhylogeny, PurgePhylogeny
import Base: ==

struct Delta{G} <: AbstractGenotype where {G <: AbstractGenotype}
    change::G 
end

struct GenePool
    deltas::Vector
end

DeltaCache = Dict{Int, Delta}

Base.:(==)(a::Delta, b::Delta) = a.change == b.change

function develop(creator::Creator, ind::Individual{I, G, D}) where {I, G <: Delta, D}
    genotype = worker_construct_child_genome(ind)
    @assert typeof(genotype) <: AbstractGenotype
    develop(creator, genotype)
end

get_tree(pop::Population) = getonly(p -> p isa PhylogeneticTree, pop.data)
get_delta_cache(pop::Population) = getonly(p -> p isa DeltaCache, pop.data)

function initialize_phylogeny!(::AbstractState, pop::Population)
    @assert isnothing(findfirst(p -> p isa PhylogeneticTree, pop.data)) "Phylogenetic tree already initialized for population $(pop.id)"
    ind_ids = [ind.id for ind in pop.individuals]
    tree = PhylogeneticTree(ind_ids)
    push!(pop.data, tree)
    # add tracking data
    io = open(phylo_fname(pop), "w")
    println(io, "id, ancestor_list")
    for ind in tree.genesis
        @assert isnothing(ind.parent)
        println(io, "$(ind.id),")
    end
    push!(pop.data, PhyloTracker(io, maximum(ind_ids)))
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
    nothing
end

@define_op "UpdatePhylogeny" # add children and prune

UpdatePhylogeny(ids::Vector{String}=String[];kwargs...) = create_op("UpdatePhylogeny",
    retriever=PopulationRetriever(ids),
    updater=map(map((s,p)->update_phylogeny!(s,p)));kwargs...)

function purge_phylogeny!(::AbstractState, pop::Population)
    pop_ids = Set(ind.id for ind in pop.individuals)
    # remove unreachable individuals
    purge_unreachable_nodes!(get_tree(pop), pop_ids)
    nothing
end

@define_op "PurgePhylogeny"
PurgePhylogeny(ids::Vector{String}=String[];kwargs...) = create_op("PurgePhylogeny",
    retriever=PopulationRetriever(ids),
    updater=map(map((s,p)->purge_phylogeny!(s,p)));kwargs...)

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

function compute_genepool(pop::Population; n_latest::Int)
    @assert pop.individuals[1].genotype isa Delta "Only delta genepools are supported"
    dc = get_delta_cache(pop)
    # ignore all individuals from current generation
    id_gens = [(ind.id, ind.generation) for ind in pop.individuals]
    # remove inds from latest generation, as they have not been selected for
    latest_gen = minimum(ig[2] for ig in id_gens)
    latest_ids = Set(ig[1] for ig in id_gens if ig[2] == latest_gen)
    older_delta_ids = setdiff(keys(dc), latest_ids) |> collect |> sort
    @assert length(older_delta_ids) >= n_latest "Not enough deltas to compute genepool, have $(length(older_delta_ids)), need $n_latest"
    older_delta_ids = older_delta_ids[end-n_latest+1:end]
    # from the older, existing deltas, compute a genepool with the n_latest deltas
    GenePool([dc[did] for did in older_delta_ids])
end

function update_genepool!(pop::Population; n_latest::Int)
    filter!(d->!isa(d, GenePool), pop.data) # get rid of any existing genepools
    push!(pop.data, compute_genepool(pop, n_latest=n_latest))
end


@define_op "UpdateGenePool"
UpdateGenePool(ids::Vector{String}=String[]; after_gen::Int, n_latest::Int, time::Bool=false, kwargs...) = 
    create_op("UpdateGenePool", 
              condition=s->generation(s) > after_gen,
              retriever=PopulationRetriever(ids),
              updater=map(map((_,p)->update_genepool!(p; n_latest=n_latest, kwargs...))),
              time=time;)

phylo_fname(pop::Population) = "$(pop.id)-phylo.csv"

mutable struct PhyloTracker <: AbstractData
    io::IO
    last_serialized::Int
end
function track_phylogeny!(pop::Population)
    pt = try 
        getonly(p -> p isa PhyloTracker, pop.data)
    catch
        @assert false "PhyloTracker not found in population $(pop.id). Make sure to call InitializePhylogeny first."
    end
    tree = get_tree(pop)
    # find and serialize inds greater than last_serialized
    for (id, node) in tree.tree
        if id > pt.last_serialized
            println(pt.io, "$id, $(node.parent.id)")
        end
    end
    flush(pt.io)
    pt.last_serialized = maximum(keys(tree.tree))
end
@define_op "TrackPhylogeny"
TrackPhylogeny(ids::Vector{String}=String[]; kwargs...) = 
    create_op("TrackPhylogeny", 
              retriever=PopulationRetriever(ids),
              updater=map(map((_,p)->track_phylogeny!(p))),
              kwargs...)
