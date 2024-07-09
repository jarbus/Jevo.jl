# Phylogeny

We use [PhylogeneticTrees.jl](https://github.com/jarbus/PhylogeneticTrees.jl) for phylogeny tracking. This package includes efficient algorithms for calculating distances between taxa and pruning the tree of taxa with no living descendants to free up memory.

## Why do we care about Phylogeny?

In short, phylogeny is the evolutionary history of a group of organisms. This has myriad uses:

* *Analysis*: We can easily observe how population dynamics change throughout evolution and visualize extinctions and sub-species. See [*Interpreting the Tape of Life* by Dolson et.al](https://lalejini.com/pubs/Dolson_et_al_2020_Interpreting_the_Tape_of_Life.pdf) for more.
* *Optimization*: We can estimate performance of individuals based on their phylogenetic distance from other individuals to achieve substantial reductions in compute required. See [evolutionary](https://arxiv.org/abs/2306.03970) and [co-evolutionary](https://arxiv.org/abs/2404.06588) applications of this idea.


## Phylogenetic Operators

Phylogeny is managed using four operators:

* [InitializePhylogeny](@ref): Adds current members of the population as roots of a phylogenetic tree. Runs on the first generation
* [UpdatePhylogeny](@ref): Updates the phylogeny for the current population, runs on all generations. Should run after children are produced.
* [LogPhylogeny](@ref): A bit misnamed for now, this operator writes phylogeny data to disk in the ALIFE Data Standard format. Should run before pruning individuals
* [PurgePhylogeny](@ref): Removes individuals from the phylogeny that have no living descendants. Should run after children are produced and optionally, all individuals have been written to disk. Essential for reducing memory usage.


## Deltas and DeltaCaches

Stores the difference between an individual and its parent for all edges in the tree.

* [InitializeDeltaCache](@ref): Initializes the delta cache for the current population. Runs on the first generation.
* [UpdateDeltaCache](@ref): Updates the delta cache for the current population. Should run after children are produced.

## Gene Pool

The [GenePool](@ref) is a subset of genes in the population, typically the most recent genes. Used for runtime techniques that leverage information about the population, like adaptive mutation rates.

* [UpdateGenePool](@ref): Creates/updates the gene pool for the current population.
