export Delta, InitializePhylogeny, UpdatePhylogeny

# TODO move this to appropriate files
struct Delta{G} <: AbstractGenotype where {G <: AbstractGenotype}
    id::Int
    parents::Vector{Int}
    change::G
end

function get_tree(pop::Population)
    trees = filter(p -> p isa PhylogeneticTree, pop.data)
    @assert length(trees) == 1 "Found $(length(trees)) phylogenetic trees for population $(pop.id)"
    tree = first(trees)
    isnothing(tree) && error("No phylogenetic tree found for population $(pop.id)")
    tree
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
    compute_pairwise_distances!(tree, pop_ids, remove_unreachable_nodes=true)
    nothing
end

@define_op "UpdatePhylogeny" # add children and prune

UpdatePhylogeny(ids::Vector{String}=String[];kwargs...) = create_op("UpdatePhylogeny",
    retriever=PopulationRetriever(ids),
    updater=map(map((s,p)->update_phylogeny!(s,p); kwargs...)))


# @define_op UpdatePhylogeny!
#
#
# """
# Pruner() 
# - prune tree at specified intervals
# - PruneDeltas() Operator
# """
# @define_op PrunePhylogeny!
#
# struct DeltaCache end # pop on master
# struct GenotypeCache end # global on worker
