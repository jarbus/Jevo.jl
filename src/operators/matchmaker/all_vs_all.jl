export AllVsAllMatchMaker
@define_op "AllVsAllMatchMaker" "AbstractMatchMaker"
AllVsAllMatchMaker(ids::Vector{String}=String[];kwargs...) =
    create_op("AllVsAllMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=make_all_v_all_matches,
          updater=add_matches!;kwargs...)


function make_all_v_all_matches(state::AbstractState, pops::Vector{Vector{Population}})
    match_counter = get_counter(AbstractMatch, state)
    env_creators = get_creators(AbstractEnvironment, state)
    @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
    env_creator = env_creators[1]
    matches = Vector{Match}()
    for i in 1:length(pops), j in i+1:length(pops) # for each pair of populations
        for subpopi in pops[i], subpopj in pops[j] # for each pair of subpopulations
            for indi in subpopi.individuals, indj in subpopj.individuals # for each pair of individuals
                push!(matches, Match(inc!(match_counter), [indi, indj], env_creator))
            end
        end
    end
    @assert length(matches) > 0
    matches
end
