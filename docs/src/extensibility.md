# Extensibility

Jevo is designed to be easily extensible. This means minimizing the number of new variants of structures a future user would need to implement. For example, with Phylogenies, we didn't need to implement a new type of population that contains a phylogenetic tree. This has a few benefits:

1. Dramatic reduction in number of types, and consequently, the lines of code.
2. Extensions work better with each other if they use the same types.
3. The user can focus on the new functionality they want to add, rather than the boilerplate code.

The way we achieve this is, frankly, by using "bad" practices very carefully, adding various checks (and documentation) to prevent misuse. So far, we've experienced fewer bugs than expected with our extension system.

## Additional Data

[States](@ref State), [Populations](@ref Population), and [Individuals](@ref Individual) can store additional data in their `.data` field, which is a vector `Any`. Operators at one point in the pipeline can write data objects to these fields, which later operators can look up use. 

For example, we add a `PhylogeneticTree` object to each population, taking care to ensure that one and only one tree exists a time. The [InitializePhylogeny](@ref) operator creates the tree at the start of a run and adds it to a population's `.data`. The [UpdatePhylogeny](@ref) operator looks up this tree and updates it after a new generation is created.

Jevo.jl provides some barebones functionality for ensuring correctness. A key function is [`getonly(f, itr::Union{Vector,Tuple})`](@ref), which will return the only object in a vector/tuple that makes `f`, return `true`. If there are no objects or more than one object that satisfies `f`, an error is thrown. In the Phylogeny system, this ensures that only one tree exists per population.
