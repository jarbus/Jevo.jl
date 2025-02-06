export BestVsBestMatchMaker

"""
    BestVsBestMatchMaker(ids::Vector{String}=String[];kwargs...)

Creates an [Operator](@ref) that creates all vs all matches between individuals in populations with ids in `ids`.
"""
@define_op "BestVsBestMatchMaker" "AbstractMatchMaker"
BestVsBestMatchMaker(ids::Vector{String}=String[];env_creator=nothing,kwargs...) =
    create_op("BestVsBestMatchMaker",
          condition=always,
          retriever=PopulationRetriever(ids),
          operator=(s,ps)->make_best_vs_best_matches(s, ps, env_creator),
          updater=add_matches!;kwargs...)


"""
    make_best_vs_best_matches(state::AbstractState, pops::Vector{Vector{Population}})

Returns a vector of [Matches](@ref Match) between all pairs of individuals in the populations.

If there is only one population, it returns the best individual in that population against itself.
"""
function make_best_vs_best_matches(state::AbstractState, pops::Vector{Vector{Population}}, env_creator)
    match_counter = get_counter(AbstractMatch, state)
    matches = Vector{Match}()

    if length(pops) == 1 && length(pops[1]) > 1
        for subpop in pops[1]
            fitnesses = [record.fitness for ind in subpop.individuals for record in ind.records]
            @assert length(fitnesses) == length(subpop.individuals)
            best_ind = subpop.individuals[argmax(fitnesses)]
            push!(matches, Match(inc!(match_counter), [best_ind, best_ind], env_creator))
        end
        return matches
    end

    for i in 1:length(pops), j in i+1:length(pops) # for each pair of populations
        for subpopi in pops[i], subpopj in pops[j] # for each pair of subpopulations
            fitnesses_i = [record.fitness for ind in subpopi.individuals for record in ind.records]
            fitnesses_j = [record.fitness for ind in subpopj.individuals for record in ind.records]
            @assert length(fitnesses_i) == length(subpopi.individuals)
            @assert length(fitnesses_j) == length(subpopj.individuals)
            best_indi = subpopi.individuals[argmax(fitnesses_i)]
            best_indj = subpopj.individuals[argmax(fitnesses_j)]
            push!(matches, Match(inc!(match_counter), [best_indi, best_indj], env_creator))
        end
    end
    @assert length(matches) > 0
    matches
end
