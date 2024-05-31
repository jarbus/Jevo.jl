# Jevo

[![Build Status](https://github.com/jarbus/Jevo.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jarbus/Jevo.jl/actions/workflows/CI.yml?query=branch%3Amaster)

# Install

This package requires modified versions of Transformers.jl and NeuralAttentionlib.jl, which are unregistered. In addition, it depends on a custom plotting library (XPlot.jl) & a library for phylogenies (PhylogeneticTrees.jl), both unregistered as well. To install all dependencies, run the following command in the environment of your choice:

```julia
julia install.jl
```

# Guidelines:

- Minimize architectural complexity, maximize code reuse
- Trade-off efficiency for simplicity whenever possible, except for performance-critical code. Don't be afraid to recompute things to avoid bookkeeping or use existing inefficient solution if it doesn't noticably impact performance.
- All Counters use the highest level appropriate type (AbstractIndividual, AbstractGene, etc)
- randn is much faster for float32 than float16, so even though it takes up more memory, we use float32 for weights
- Any multi-threaded operation that uses RNG should generate a new RNG object for each iteration/thread in the main process, to ensure that the same random numbers are not used in different threads
- All evaluations are done solely on workers, and evolutionary operations are done on the main process. A worker can be on the same machine as the main process, or on a different machine.


# Design of population operators

- population retriever gets all subpopulations for each pop id provided
- selector filters individuals 
- reproducer creates individuals them
- mutator replaces the children with new individuals
- need a retriever to get all children and an update to update all children

# Design of reporters
logging to txt, writing to hdf5
for distributed, what if you launch new workers each gen if one of them dies?
each gen send the children and reconstruct from the cache?

# Design of phylo
- we store phylo tree and delta cache with population, because those are NOT LRU
- we store genotype cache and weight cache outside of population because those ARE LRU

# Design of gene pool
