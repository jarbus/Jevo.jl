export SoloMatchMaker
@define_op "SoloMatchMaker" "AbstractMatchMaker"
SoloMatchMaker(ids::Vector{String}=String[];kwargs...) =
    create_op("SoloMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=make_solo_matches,
          updater=add_matches!;kwargs...)


function make_solo_matches(state::AbstractState, pops::Vector{Vector{Population}})
    match_counter = get_counter(AbstractMatch, state)
    env_creators = get_creators(AbstractEnvironment, state)
    @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
    env_creator = env_creators[1]
    matches = Vector{Match}()
    pop = pops[1][1]
    for subpop in pops, pop in subpop, ind in pop.individuals
        push!(matches, Match(inc!(match_counter), [ind], env_creator))
    end
    matches
end
