export FileSystemMatchMaker
"""
"""
@define_op "FileSystemMatchMaker" "AbstractMatchMaker"
FileSystemMatchMaker(match_queue_dir::String, ids::Vector{String}=String[]; env_creator=nothing, kwargs...) =
    create_op("FileSystemMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=(s,ps)->make_filesystem_match(s, ps, match_queue_dir, env_creator),
          updater=add_matches!;kwargs...)


function make_filesystem_match(state::AbstractState, pops::Vector{Vector{Population}}, match_queue_dir::String, env_creator)
    match_counter = get_counter(AbstractMatch, state)
    if isnothing(env_creator)
        env_creators = get_creators(AbstractEnvironment, state)
        @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
        env_creator = env_creators[1]
    end
    @assert !isnothing(env_creator) "Environment creator must be defined"
    matches = Vector{Match}()
    # read match_queue from local dir
    matches = deserialize(match_queue_dir)
    # make matches
    for match in matches
        inds = Vector(undef, length(match))
        inserted = fill(false, length(match))
        for subpop in pops, pop in subpop, ind in pop.individuals, (idx, id) in enumerate(match)
            if id == ind.id
                inds[idx] = ind
                inserted[idx] = true
            end
        end
        for (idx, id) in enumerate(match)
            if id == -1
                inds[idx] = Individual(-1, generation(state), Int[], FSGenotype, FSPhenotype)
                inserted[idx] = true
            end
        end

        @assert all(inserted)
        @assert count(ind->ind.id == -1, inds) == 1
        push!(matches, Match(inc!(match_counter), inds, env_creator))
    end
    matches
end
