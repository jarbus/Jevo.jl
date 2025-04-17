export FileSystemMatchMaker

@define_op "FileSystemMatchMaker" "AbstractMatchMaker"
FileSystemMatchMaker(match_queue_dir::String, ids::Vector{String}=String[]; env_creator=nothing, null_action_space::Tuple{Vararg{Int}}, kwargs...) =
    create_op("FileSystemMatchMaker",
          condition=always,
          retriever=get_individuals(ids),
          operator=(s,ps)->make_filesystem_match(s, ps, match_queue_dir, env_creator, null_action_space),
          updater=add_matches!;kwargs...)


function make_filesystem_match(state::AbstractState, individuals::Vector{Individual}, match_queue_dir::String, env_creator, null_action_space::Tuple{Vararg{Int}})
    match_counter = get_counter(AbstractMatch, state)
    if isnothing(env_creator)
        env_creators = get_creators(AbstractEnvironment, state)
        @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
        env_creator = env_creators[1]
    end
    @assert !isnothing(env_creator) "Environment creator must be defined"
    matches = Vector{Match}()
    # make a file containing all individual.ids
    open(joinpath(match_queue_dir, "available_ids.jls"), "w") do io
        serialize(io, [ind.id for ind in individuals])
    end

    # read match_queue from local dir
    matches_file = joinpath(match_queue_dir, "queue.jls")
    # Poll filesystem until queue.jls exists
    while !isfile(matches_file)
        @info "Polling $matches_file..."
        sleep(1)
    end
    @info "$matches_file found"
    matches = isfile(matches_file) ? deserialize(matches_file) : Vector{Vector{Int}}()
    # remove match_queue_dir/queue
    isfile(matches_file) && rm(matches_file)

    # make matches
    for match in matches
        inds = Vector(undef, length(match))
        inserted = fill(false, length(match))
        for ind in individuals, (idx, id) in enumerate(match)
            if id == ind.id
                inds[idx] = ind
                inserted[idx] = true
            end
        end
        for (idx, id) in enumerate(match)
            if id == -1
		    inds[idx] = Individual(-1, generation(state), Int[], Creator(FSGenotype, (dirpath=match_queue_dir, null_action_space=null_action_space)), Creator(FSPhenotype))
                inserted[idx] = true
            end
        end

        @assert all(inserted)
        @assert count(ind->ind.id == -1, inds) == 1
        @info
        push!(matches, Match(inc!(match_counter), inds, env_creator))
    end
    matches
end
