export CreateMissingWorkers 
"""Creates up to `n` workers"""
@define_op "CreateMissingWorkers"
CreateMissingWorkers(n=Int;kwargs...) = create_op("CreateMissingWorkers",
          condition=always,
          operator=(_,_)->create_missing_workers(n);
          kwargs...)

function create_missing_workers(n::Int)
    n_workers = length(workers())
    n_workers_to_add = n - n_workers
    addprocs(SlurmManager(n_workers_to_add), gres="gpu:1", c=4)
    @everywhere include(joinpath(@__DIR__, "load_jevo.jl"))
end
