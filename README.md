# Jevo

[![Build Status](https://github.com/jarbus/Jevo.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jarbus/Jevo.jl/actions/workflows/CI.yml?query=branch%3Amaster)

# WARNING: DO NOT USE CROSSOVER, WE ARE CACHING WEIGHTS BASED ON THE LAST GENE ID, CROSSOVER of GENE IDS CAN BREAK THIS

# Guidelines:

- Minimize architectural complexity, maximize code reuse
- All Counters use the highest level appropriate type (AbstractIndividual, AbstractGene, etc)
- randn is much faster for float32 than float16, so even though it takes up more memory, we use float32 for weights
- Any multi-threaded operation that uses RNG should generate a new RNG object for each iteration/thread in the main process, to ensure that the same random numbers are not used in different threads

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
