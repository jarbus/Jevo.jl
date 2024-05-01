export CreateMissingWorkers 
"""Creates up to `n` workers"""
@define_op "CreateMissingWorkers"
CreateMissingWorkers(n=Int;kwargs...) = create_op("CreateMissingWorkers",
          condition=always,
          operator=(_,_)->create_missing_workers(n);
          kwargs...)

function create_missing_workers(n::Int)
    n_tries = 0
    while length(workers()) < n
        println("Creating worker")
        addprocs(SlurmManager(1))
        n_tries += 1
        n_tries >= 20 && error("Too many attempts to create missing workers")
    end
    @everywhere include(joinpath(@__DIR__, "load_jevo.jl"))
end
