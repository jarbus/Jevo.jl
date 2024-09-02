export CreateMissingWorkers 
"""
    CreateMissingWorkers(n=Int; slurm::Bool=true, c::Int, kwargs...)

Creates up to `n` workers, each with `c` processors.
"""
@define_op "CreateMissingWorkers"
CreateMissingWorkers(n=Int; slurm::Bool=true, c::Int, kwargs...) = create_op("CreateMissingWorkers",
          condition=always,
          operator=(_,_)->create_missing_workers(n,slurm=slurm, c=c);
          kwargs...)

function create_missing_workers(n::Int; slurm::Bool, c::Int)
    n_workers = workers()[1] == 1 ? 0 : length(workers())
    n_workers_to_add = n - n_workers
    if n_workers_to_add > 0
        if slurm 
            addprocs(SlurmManager(n_workers_to_add), gres="gpu:$n", c=c)
        else
            addprocs(n_workers_to_add)
        end
        @everywhere include(joinpath(@__DIR__, "load_jevo.jl"))
    end
end
