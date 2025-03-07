export BestVsAllMatchMaker

@define_op "BestVsAllMatchMaker" "AbstractMatchMaker"
BestVsAllMatchMaker(ids::Vector{String}=String[];env_creator=nothing,kwargs...) =
    create_op("BestVsAllMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=(s,ps)->make_best_vs_all_matches(s, ps, env_creator),
          updater=add_matches!;kwargs...)


function make_best_vs_all_matches(state::AbstractState, pops::Vector{Vector{Population}}, env_creator)
    match_counter = get_counter(AbstractMatch, state)
    matches = Vector{Match}()


    if length(pops) == 1 && length(pops[1]) == 1
        pop = pops[1][1]
        fitnesses = [record.fitness for ind in pop.individuals for record in ind.records]
        @assert length(fitnesses) == length(pop.individuals)
        best_ind = pop.individuals[argmax(fitnesses)]

        for ind in pop.individuals
            if ind != best_ind
                push!(matches, Match(inc!(match_counter), [best_ind, ind], env_creator))
                push!(matches, Match(inc!(match_counter), [ind, best_ind], env_creator))
            else
                push!(matches, Match(inc!(match_counter), [ind, ind], env_creator))
            end
        end
        @assert length(matches) == 2 * length(pop.individuals) - 1
    else
        @error "BestVsAllMatchMaker: only one population is allowed for now"
    end
    matches
end
