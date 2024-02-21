export AllVsAllMatchMaker, make_matches!
struct AllVsAllMatchMaker <: AbstractMatchMaker
    condition::Function
    retriever::AbstractRetriever
    operator::Function
    updater::Union{AbstractUpdater, Function}
    data::Vector{AbstractData}
end

function AllVsAllMatchMaker(ids::Vector{String}=String[])
    condition = always
    retriever = PopulationRetriever(ids)
    operator = noop
    updater = make_matches!
    AllVsAllMatchMaker(condition, retriever, operator, updater, AbstractData[])
end


function make_matches!(state::AbstractState, pops::Vector{Vector{Population}})
    match_counter = get_counter(AbstractMatch, state)
    env_creators = get_creators(AbstractEnvironment, state)
    @assert length(env_creators) == 1 "There should be exactly one environment creator for the time being, found $(length(env_creators))."
    env_creator = env_creators[1]
    for i in 1:length(pops), j in i+1:length(pops) # for each pair of populations
        for subpopi in pops[i], subpopj in pops[j] # for each pair of subpopulations
            for indi in subpopi.individuals, indj in subpopj.individuals # for each pair of individuals
                push!(state.matches, Match(inc!(match_counter), [indi, indj], env_creator))
            end
        end
    end
    @assert length(state.matches) > 0
end
