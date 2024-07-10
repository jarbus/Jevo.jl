# Operators

Each step of an evolutionary algorithm is represented by an [Operator](@ref). The [@define_op](@ref) macro defines new operator *structs* with fields specified in the [operator documentation](@ref Operator). The [create_op](@ref) function generates new operator *objects* with default values for unspecified operator fields. This section provides an overview of the operators implemented so far in Jevo.jl.

### Retrievers

Retrievers are a struct or function that, retrieve data from the state. 

* [Jevo.PopulationCreatorRetriever](@ref), used in [InitializeAllPopulations](@ref)
* [PopulationRetriever](@ref), used in [AllVsAllMatchMaker](@ref), and many others
* [get_individuals](@ref), used in [ClearInteractionsAndRecords](@ref)

### Updaters

* [Jevo.ComputeInteractions!](@ref), used in [Performer](@ref)
* [PopulationAdder](@ref), used in [InitializeAllPopulations](@ref)
* [PopulationUpdater](@ref), currently unused because I forgot to use it 
* [Jevo.add_matches!](@ref), used in [AllVsAllMatchMaker](@ref) and [SoloMatchMaker](@ref)
* [Jevo.RecordAdder](@ref), used in [ScalarFitnessEvaluator](@ref)

### Matchmaker

* [AllVsAllMatchMaker](@ref)
* [SoloMatchMaker](@ref), individuals play a match alone, used for evolutionary computing

### Evaluators

* [ScalarFitnessEvaluator](@ref)
* [RandomEvaluator](@ref)

### Selectors

* [TruncationSelector](@ref)

### Reproducers

* [CloneUniformReproducer](@ref)

### Performer

* [Performer](@ref)


### Mutators

* [Mutator](@ref), uses [Jevo.mutate](@ref) as its `.operator`.

### Assertors

Assertors are operators that you can add at any point in the pipeline to check that certain aspects of the state are as expected.

* [PopSizeAssertor](@ref)

### Reporters

* [Reporter](@ref), can log data if [`Jevo.measure`](@ref) for a specified [Jevo.AbstractMetric](@ref) as its `.operator`.

### Checkpointer

* [Checkpointer](@ref)

### Initializers

* [InitializeAllPopulations](@ref), uses [Jevo.create](@ref) as its `.operator`.

### Miscellaneous

* [CreateMissingWorkers](@ref). SLURM compatible, but only for a single node.

### Phylogenies

* See [Phylogeny](@ref)
