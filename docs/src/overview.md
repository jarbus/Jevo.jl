# Overview

Jevo is designed to be simple yet flexible, with the core package under 2k LOC. The framework is broken up into four core concepts:

1. A [State](@ref) is a structure that holds all information about the evolutionary process.
2. An [Operator](@ref) is a function/struct that updates the state. Jevo breaks down evolutionary algorithms into a series of sequential operators which are applied to the simulator state.
3. A [Creator](@ref) is a structure that generates new objects with specified parameters, and are called by operators. Creators spawn populations, individuals, genotypes, environments, other creators, and more.
4. All other standard evolutionary objects ([`Individual`](@ref), [`Population`](@ref), genotypes, phenotypes, environments, etc.) which are contained in the [`State`](@ref).

```@docs
State
```

```@docs
Creator
PassThrough
```

```@docs
Individual
Population
CompositePopulation
```
