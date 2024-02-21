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
