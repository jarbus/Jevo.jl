# Jevo

[![Build Status](https://github.com/jarbus/Jevo.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jarbus/Jevo.jl/actions/workflows/CI.yml?query=branch%3Amaster)

# Guidelines:

- Minimize architectural complexity, maximize code reuse
- All Counters use the highest level appropriate type (AbstractIndividual, AbstractGene, etc)

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
