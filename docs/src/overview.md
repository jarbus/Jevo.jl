# Overview

Jevo is designed to be simple yet flexible (co-)evolutionary computing framework, with the core package under 2500 LOC. The framework is broken up into four core concepts:

1. A [State](@ref) is a structure that holds all information about the evolutionary process.
2. [Operators](@ref Jevo.Operator) are functions/structs that update the state. Jevo breaks down evolutionary algorithms into a series of sequential operators which are applied to the simulator state.
3. A [Creator](@ref) is a structure that generates new objects with specified parameters, and are called by operators. Creators spawn populations, individuals, genotypes, environments, other creators, and more.
4. All other standard evolutionary objects ([`Individual`](@ref), [`Population`](@ref), genotypes, phenotypes, environments, etc.) which are contained in the [`State`](@ref).

## Numbers Game Example

The simplest way to get started with Jevo is by example. We'll use [The Numbers Game](http://www.demo.cs.brandeis.edu/papers/gecco2001_cdms.pdf) as our example domain, which consists of two co-evolving populations of vectors. For an example which focuses on neuroevolution, see the TODO example.

```julia
using Jevo
using Logging
using StableRNGs

# Jevo provides a custom logger to easily store statistical measurements to 
# an HDF5 file, print to console, and log text to a file.
global_logger(JevoLogger())
rng = StableRNG(1)

k = 2          # How many individuals to keep each generation
n_dims = 2     # Number of dimensions in the vector
n_inds = 10    # Number of individuals in each population
n_species = 2  # Number of species
n_gens = 10    # Number of generations

# We create a list of counters which are incremented 
# to track individuals, genes, generations, and matches.
# This is passed to the state constructor.
counters = default_counters()

# Instead of creating genotypes and phenotypes directly, we create a
# genotype creator, which generates genotypes, and a phenotype creator,
# which "develops" a genotype into a phenotype.
ng_genotype_creator = Creator(VectorGenotype, (n=n_dims,rng=rng))
ng_developer = Creator(VectorPhenotype)

# Likewise, instead of creating populations directly, we create a population
# creator, which generates populations. Here, we create a composite population,
# which is a population of sub-populations. Each sub-population is a species.
comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_genotype_creator, ng_developer) for i in 1:n_species], counters))

# An environment creator can be used to generate instances of an environment,
# particulary useful for randomizing the environment.
env_creator = Creator(CompareOnOne)

# We create a state called "ng_phylogeny" with the RNG object we passed to our creators.
# We initialize the state with a list of creators (order does not matter), and
# a list of operators (order does matter). Operators will look for creators by
# type when needed to generate new objects when appropriate.
state = State("ng_phylogeny", rng,[comp_pop_creator, env_creator],
    [InitializeAllPopulations(),
     InitializePhylogeny(),
    AllVsAllMatchMaker(),
    Performer(),
    ScalarFitnessEvaluator(),
    TruncationSelector(k),
    CloneUniformReproducer(n_inds),
    Mutator(),
    PopSizeAssertor(n_inds),
    ClearInteractionsAndRecords(),
    Reporter(GenotypeSum, console=true)], counters=counters)

# We run the state for n_gens generations.
run!(state, n_gens)
```

## Design Philosophy

- Minimize architectural complexity, maximize code reuse.
- Trade-off efficiency for simplicity whenever possible, **except for performance-critical code**. Don't be afraid to recompute things to avoid bookkeeping or use existing inefficient solution if it doesn't noticably impact performance.
- All [`Counters`](@ref Counter) use the highest level appropriate type (`AbstractIndividual`, `AbstractGene`, etc).
- `randn` is faster for `Float32` than `Float16`, so we use `Float32` for weights, despite the added memory cost.
- Any multi-threaded operation that uses `state.rng` should generate a new RNG object for each iteration or thread in the main process, to ensure that the same random numbers are not used in different threads.
- When using Distributed, all evaluations are done on workers and other evolutionary operations are done on the main process. When not using Distributed, all operations are done on the main process.
