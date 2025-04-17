export FileSystemGenotype, FileSystemPhenotype, develop, mutate, FileSystemMatchMaker

struct FileSystemGenotype <: AbstractGenotype
    dirpath::String
    null_action_space::Tuple{Vararg{Int}}

end
struct FileSystemPhenotype <: AbstractPhenotype
    dirpath::String
    null_action_space::Tuple{Vararg{Int}}
    run_until_done::Bool
end

function mutate(::AbstractRNG, ::AbstractState, ::AbstractPopulation, genotype::FileSystemGenotype)
    @error "mutate() not defined for FileSystemGenotype"
end

develop(::Creator, genotype::FileSystemGenotype) = FileSystemPhenotype(genotype.dirpath, genotype.null_action_space, false)

function (fsphenotype::FileSystemPhenotype)(observation)
    if fsphenotype.run_until_done
        return zeros(Float32, fsphenotype.null_action_space)
    end
    # serialize to obs.jls
    open(joinpath(fsphenotype.dirpath, "observation.jls"), "w") do io
        serialize(io, observation)
    end

    # poll until action.jls exists, then deserialize and return it
    actionfile = joinpath(fsphenotype.dirpath, "action.jls")
    while !isfile(actionfile)
        sleep(0.3)
    end

    action = open(actionfile, "r") do io
        deserialize(io)
    end
    rm(actionfile)
    action
end


@define_op "FileSystemMatchMaker" "AbstractMatchMaker"
FileSystemMatchMaker(match_queue_dir::String; env_creator=nothing, null_action_space::Tuple{Vararg{Int}}, kwargs...) =
    create_op("FileSystemMatchMaker",
          condition=always,
          retriever=get_individuals,
          operator=(s,is)->make_filesystem_match(s, is, match_queue_dir, env_creator, null_action_space),
          updater=add_matches!;kwargs...)


function make_filesystem_match(state::AbstractState, individuals::Vector{I}, match_queue_dir::String, env_creator, null_action_space::Tuple{Vararg{Int}}) where {I<:AbstractIndividual}
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
        inds = Vector{I}(undef, length(match))
        inserted = fill(false, length(match))
        @info "Making Match"
        for ind in individuals, (idx, id) in enumerate(match)
            if id == ind.id
                @info "Found $id in match"
                inds[idx] = ind
                inserted[idx] = true
            end
        end
        for (idx, id) in enumerate(match)
            if id == -1
                @info "Found FileSystemPlayer"
		        inds[idx] = Individual(-1, generation(state), Int[], FileSystemGenotype(match_queue_dir, null_action_space), Creator(FileSystemPhenotype))
                inserted[idx] = true
            end
        end

        @assert all(inserted)
        @assert count(ind->ind.id == -1, inds) == 1
        push!(matches, Match(inc!(match_counter), inds, env_creator))
    end
    matches
end



