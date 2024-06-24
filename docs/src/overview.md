# Overview

Jevo is designed to be simple yet flexible, with the core package under 2k LOC. The framework is broken up into four core concepts:

1. A [State](@ref) is a structure that holds all information about the evolutionary process.
2. [Operator](@ref)s are functions/structs that update the state. Jevo breaks down evolutionary algorithms into a series of sequential operators which are applied to the simulator state.
3. A [Creator](@ref) is a structure that generates new objects with specified parameters, and are called by operators. Creators spawn populations, individuals, genotypes, environments, other creators, and more.
4. All other standard evolutionary objects ([`Individual`](@ref), [`Population`](@ref), genotypes, phenotypes, environments, etc.) which are contained in the [`State`](@ref).

## State

```@docs
State
State(rng::AbstractRNG, creators::Vector{<:AbstractCreator}, operators::Vector{<:AbstractOperator})
run!
```


```@docs
Creator
PassThrough
```

## Design Philosophy


## Guidelines:

- Minimize architectural complexity, maximize code reuse.
- Trade-off efficiency for simplicity whenever possible, except for performance-critical code. 
    - Don't be afraid to recompute things to avoid bookkeeping or use existing inefficient solution if it doesn't noticably impact performance.
- All Counters use the highest level appropriate type (AbstractIndividual, AbstractGene, etc).
- randn is faster for float32 than float16, so we use float32 for weights, despite the added memory cost.
- Any multi-threaded operation that uses `state.rng` should generate a new RNG object for each iteration/thread in the main process, to ensure that the same random numbers are not used in different threads.
- When using Distributed, all evaluations are done on workers and evolutionary operations are done on the main process.
