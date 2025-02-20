export UpdateParentsAcrossAllWorkers
GPID_PID_PD = Tuple{Int, Int, Union{Delta, Nothing}}
MISSING_PARENTS_PER_WORKER = Dict{Int, Vector{Int}}  # worker_id: [pids]
@define_op "UpdateParentsAcrossAllWorkers"

UpdateParentsAcrossAllWorkers(ids::Vector{String}=String[];kwargs...) = create_op("UpdateParentsAcrossAllWorkers",
    retriever=PopulationRetriever(ids),
    updater=(s, ps)->update_parents_across_all_workers!(s, ps); kwargs...)

function update_parents_across_all_workers!(s::State, pops::Vector{Vector{Population}})
    # all these functions run on master, and make calls to workers
    # check which workers are missing parents of the current generation
    workers_missing_parents = master_send_pids_and_gpids(pops)
    # construct all missing parents on master 
    worker_pid_genomes = master_construct_parents_genomes(pops, workers_missing_parents)
    # TODO for any worker that's missing a genome, we also want to re-transmit the weight cache
    # send to workers and cache
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
    else  # we encode genesis individuals as pid=-1, gpid=id
        gpid, pid, parent_delta = ind.id, -1, dc[ind.id]
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
    any(!isempty, values(workers_missing_parents)) && @info workers_missing_parents
    return workers_missing_parents
end

function worker_mk_parents_from_deltas_and_ret_missing!(gpid_pid_pds::Vector{GPID_PID_PD})
    miss, geno_cache = Int[], get_genotype_cache()
    for (gpid, pid, pd) in gpid_pid_pds
        if pid == -1 && gpid == -1
            error("Found an individual with no parent and no grandparent, this used to be valid for all genesis inds, but now we encode geneis as pid=-1, gpid=id")
        elseif pid == -1 && gpid != -1 
            # we encode an org with no parent by it's grandparent id
            # this is only for the first generation
            geno_cache[gpid] = deepcopy(pd.change)
        elseif gpid != -1 && gpid ∈ keys(geno_cache)
            geno_cache[pid] = geno_cache[gpid] + pd
        else
            push!(miss, pid)
        end
    end
    unique(miss)
end

master_construct_genome(ind::Individual, pop::Population) = 
    master_construct_genome(ind, get_tree(pop), get_delta_cache(pop), get_genotype_cache())
master_construct_genome(ind::Individual, tree::PhylogeneticTree, dc::DeltaCache, gc::_GenotypeCache) =
    master_construct_genome(ind.id, tree, dc, gc)

function master_construct_genome(id::Int, tree::PhylogeneticTree, dc::DeltaCache, gc::_GenotypeCache)
    id ∈ keys(gc) && return gc[id]
    # go back up the tree to find the nearest cached ancestor in genotype cache
    # then construct the genome from the nearest ancestor to the individual
    # by applying deltas
    path = [tree.tree[id]] # start path from individual to nearest ancestor
    # go up the tree until we find a cached ancestor
    while !isnothing(path[end].parent) && !(path[end].parent.id in keys(gc))
        push!(path, path[end].parent)
    end 
    # restore from cache if there's an extant parent, otherwise, start from last node
    parent = path[end].parent
    genome = if !isnothing(parent) && parent.id ∈ keys(gc)
        gc[parent.id]
    else
        dc[pop!(path).id].change
    end
    # apply deltas from top of tree to bottom to construct genome
    for node in reverse(path)
        genome += dc[node.id]
    end
    genome
end


function worker_construct_child_genome(ind::Individual{G, D, I}) where {G <: Delta, D, I}
    @assert length(ind.parents) <= 1
    gc = get_genotype_cache()
    ind.id ∈ keys(gc) && return gc[ind.id]
    length(ind.parents) == 0 && return ind.genotype.change
    @assert ind.parents[1] ∈ keys(gc) "parent $(ind.parents[1]) not found on process $(myid()), keys=$(keys(gc))"
    #genotype = gc[ind.parents[1]] + ind.genotype
    genotype = add_delta_to_genome(gc[ind.parents[1]], ind.genotype)
    genotype
end

function master_construct_parents_genomes(pops::Vector{Vector{Population}}, workers_missing_parents::MISSING_PARENTS_PER_WORKER)
    pids = reduce(union!, values(workers_missing_parents), init=Set{Int}())
    gc = get_genotype_cache()
    parent_genomes = Dict(pid=>gc[pid] for pid in pids if pid in keys(gc))
    missing_pids = setdiff(pids, keys(parent_genomes))
    for comp_pop in pops, subpop in comp_pop
        tree, dc = get_tree(subpop), get_delta_cache(subpop)
        for id in missing_pids
            if id ∈ keys(tree.tree)
                @assert id ∈ keys(dc) "id $(id) found in tree but not found in delta cache"
                gc[id] = parent_genomes[id] = master_construct_genome(id, tree, dc, gc)
                pop!(missing_pids, id)
            end
        end
        #= for ind in subpop.individuals =#
        #=     # send parents if they currently exist in the population =#
        #=     if ind.id in pids && !(ind.id in keys(parent_genomes)) =#
        #=         gc[ind.id] = parent_genomes[ind.id] = master_construct_genome(ind, tree, dc, gc) =#
        #=     end =#
        #= end =#

    end
    @assert isempty(missing_pids) "missing $(missing_pids) when constructing parent genomes"
    @assert pids ⊆ keys(parent_genomes) "missing $(setdiff(pids, keys(parent_genomes))) for child pids $pids given parent pids $(keys(parent_genomes)). Is your genotype cache too small?"
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
