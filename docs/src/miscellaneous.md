# Miscellaneous Bits

## Checkpointing

The [Checkpointer](@ref) operator serializes the state to disk, and the [LoadCheckpoint](@ref) operator loads it. 

To enable checkpointing with this operator, set the `checkpoint_interval` kwarg in your state creation logic:

```julia
state = State("example", rng, creators::Vector{<:AbstractCreator}, 
            [LoadCheckpoint(),
            # other operators...
            ], counters=counters, checkpoint_interval=100
# Run! exists when we load a checkpoint, so we call it again.
# Run! will always terminate if state is greater than the specified generation,
#   so there is no risk of running 2k generations.
# this
run!(state, 1000)
run!(state, 1000)
```

## SLURM

Jevo.jl supports distributed computing on [SLURM](https://slurm.schedmd.com/overview.html) clusters. Jevo currently only supports GPU workers on a single node, but will support distributed computing across nodes in the future.
