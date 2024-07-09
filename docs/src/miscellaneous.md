# Miscellaneous Bits

## Checkpointing

The [Checkpointer](@ref) operator serializes the state to disk. As the state contains every part of an evolutionary simulation (except per-worker caches), this is sufficient to resume from a checkpoint.

For checkpointing, you can use the following pattern for state creation:

```julia
checkpointname = "check.jls"
state = isfile(checkpointname) ? restore_from_checkpoint(checkpointname) :
          State("example", rng, creators::Vector{<:AbstractCreator}, 
            [Checkpointer(checkpointname, interval=25),
            # other operators...
            ]
```

## SLURM

Jevo.jl supports distributed computing on [SLURM](https://slurm.schedmd.com/overview.html) clusters. Jevo currently only supports GPU workers on a single node, but will support distributed computing across nodes in the future.

[CreateMissingWorkers
