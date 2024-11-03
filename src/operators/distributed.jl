export CreateMissingWorkers 
"""
    CreateMissingWorkers(n=Int; slurm::Bool=true, c::Int, kwargs...)

Creates up to `n` workers, each with `c` processors.
"""
@define_op "CreateMissingWorkers"
CreateMissingWorkers(n=Int; n_gpus::Int=0, slurm::Bool=true, c::Int, kwargs...) = create_op("CreateMissingWorkers",
          condition=always,
          operator=(_,_)->create_missing_workers(n,slurm=slurm, c=c, n_gpus=n_gpus);
          kwargs...)

function create_missing_workers(n::Int; slurm::Bool, c::Int, n_gpus::Int)
    n_workers = workers()[1] == 1 ? 0 : length(workers())
    n_workers_to_add = n - n_workers
    if n_workers_to_add > 0
        if slurm 
            @info("adding $n_workers_to_add with $n_gpus")
            addprocs(SlurmManager(n_workers_to_add), c=c, env=["JULIA_CUDA_SOFT_MEMORY_LIMIT"=>ENV["JULIA_CUDA_SOFT_MEMORY_LIMIT"]])
        else
            addprocs(n_workers_to_add)
        end
        @everywhere include(joinpath(@__DIR__, "load_jevo.jl"))
    end
end
