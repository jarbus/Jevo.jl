export UpdateParentsAcrossAllWorkers
GPID_PID_PD = Tuple{Int, Int, Union{Delta, Nothing}}
MISSING_PARENTS_PER_WORKER = Dict{Int, Vector{Int}}  # worker_id: [pids]
@define_op "UpdateParentsAcrossAllWorkers"

UpdateParentsAcrossAllWorkers(ids::Vector{String}=String[];kwargs...) = create_op("UpdateParentsAcrossAllWorkers",
    retriever=PopulationRetriever(ids),
    updater=(s, ps)->update_parents_across_all_workers!(s, ps); kwargs...)

function update_parents_across_all_workers!(s::State, pops::Vector{Vector{Population}})
    # all these functions run on master, and make calls to workers
    workers_missing_parents = master_send_pids_and_gpids(pops)
    worker_pid_genomes = master_construct_parents_genomes(pops, workers_missing_parents)
    master_cache_parents!(worker_pid_genomes)
end


function master_get_gpid_pid_pds(ind::Individual, tree::PhylogeneticTree, dc::DeltaCache)
    pid, gpid = -1, -1
    parent_node, grandparent, parent_delta = tree.tree[ind.id].parent, nothing, nothing
    if !isnothing(parent_node) 
        pid = parent_node.id
        parent_delta = dc[pid]
        grandparent = tree.tree[pid].parent
        if !isnothing(grandparent) 
            gpid = grandparent.id
        end
    end
    gpid, pid, parent_delta
end

function master_send_pids_and_gpids(pops::Vector{Vector{Population}})
    gpid_pid_pds = Vector{GPID_PID_PD}()
    # First, compute (grandparent id, parent id, parent delta) pairs
    for comp_pop in pops
        for subpop in comp_pop
            tree, dc = get_tree(subpop), get_delta_cache(subpop)
            for ind in subpop.individuals
                push!(gpid_pid_pds, master_get_gpid_pid_pds(ind, tree, dc))
            end
        end
    end
    gpid_pid_pds = unique(gpid_pid_pds)
    # Send to workers
    tasks = [@spawnat wid worker_mk_parents_from_deltas_and_ret_missing!(gpid_pid_pds)
             for wid in procs()]
    # Receive missing parents from workers 
    workers_missing_parents = Dict(task.where => fetch(task) for task in tasks)
    @info workers_missing_parents
    return workers_missing_parents
end

function worker_mk_parents_from_deltas_and_ret_missing!(gpid_pid_pds::Vector{GPID_PID_PD})
    miss, geno_cache = Int[], get_genotype_cache()
    for (gpid, pid, pd) in gpid_pid_pds
        pid == -1 && continue
        if gpid != -1 && gpid ∈ keys(geno_cache)
            geno_cache[pid] = geno_cache[gpid] + pd
        else
            push!(miss, pid)
        end
    end
    unique(miss)
end

function master_construct_genome(ind::Individual, tree::PhylogeneticTree, dc::DeltaCache, gc::_GenotypeCache)
    # go back up the tree to find the nearest cached ancestor in genotype cache
    # then construct the genome from the nearest ancestor to the individual
    # by applying deltas
    path = [tree.tree[ind.id]] # start path from individual to nearest ancestor
    # go up the tree until we find a cached ancestor
    while !isnothing(path[end].parent) && !(path[end].parent.id in keys(gc))
        push!(path, path[end].parent)
    end 
    # restore from cache if there's an extant parent, otherwise, start from last node
    genome = !isnothing(path[end].parent) ? gc[path[end].parent.id] : dc[pop!(path).id].change
    # apply deltas from top of tree to bottom to construct genome
    for node in path
        genome += dc[node.id]
    end
    genome
end

function worker_construct_child_genome(ind::Individual{I, G, D}) where {I, G <: Delta, D}
    @assert length(ind.parents) <= 1
    gc = get_genotype_cache()
    length(ind.parents) == 0 && return ind.genotype.change
    @assert ind.parents[1] ∈ keys(gc) "parent $(ind.parents[1]) not found on process $(myid())"
    genotype = gc[ind.parents[1]] + ind.genotype
    genotype
end

function master_construct_parents_genomes(pops::Vector{Vector{Population}}, workers_missing_parents::MISSING_PARENTS_PER_WORKER)
    pids = reduce(union!, values(workers_missing_parents), init=Set{Int}())
    gc = get_genotype_cache()
    parent_genomes = Dict(pid=>gc[pid] for pid in pids if pid in keys(gc))
    for comp_pop in pops, subpop in comp_pop
        tree, dc = get_tree(subpop), get_delta_cache(subpop)
        for ind in subpop.individuals
            if ind.id in pids && !(ind.id in keys(parent_genomes))
                parent_genomes[ind.id] = master_construct_genome(ind, tree, dc, gc)
            end
        end
    end
    @assert all(pid->pid ∈ keys(parent_genomes), pids)
    worker_parent_genomes = Dict(wid =>[(pid, parent_genomes[pid]) for pid in pids]
                                 for (wid, pids) in workers_missing_parents)
    return worker_parent_genomes
end

function worker_cache_parents!(pids_genomes)
    gc = get_genotype_cache()
    for (pid, genome) in pids_genomes
        gc[pid] = genome
    end
end

function master_cache_parents!(worker_parent_genomes::Dict{Int, <:Any})
    # add tasks
    tasks = [@spawnat wid worker_cache_parents!(ids_genomes)
             for (wid, ids_genomes) in worker_parent_genomes]
    # wait for tasks
    Threads.@threads for i in eachindex(tasks)
        fetch(tasks[i])
    end
end
