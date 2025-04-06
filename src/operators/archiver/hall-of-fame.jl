export AddToHallOfFame, SampleHallOfFameIntoPopulation 

struct HallOfFame
    individuals::Vector
end


@define_op "AddToHallOfFame" 
@define_op "SampleHallOfFameIntoPopulation"

AddToHallOfFame(maxsize::Int, ids::Vector{String}=String[];kwargs...) =
    create_op("AddToHallOfFame",
          retriever=PopulationRetriever(ids),
          updater=(s, ps)->add_to_hall_of_fame!(s, ps, maxsize),
          ;kwargs...)

SampleHallOfFameIntoPopulation(n::Int, ids::Vector{String}=String[];  kwargs...) =
    create_op("SampleHallOfFameIntoPopulation",
          retriever=PopulationRetriever(ids),
          updater=(s, ps)->sample_hall_of_fame_into_population!(s, ps, n),
          ;kwargs...)

function add_to_hall_of_fame!(::AbstractState, pops::Vector{Vector{Population}}, maxsize::Int)
    if length(pops) != 1 || length(pops[1]) != 1
        @error "hall of fame not implemented for multiple populations yet"
    end
    pop = pops[1][1]
    hofs = filter(x->x isa HallOfFame, pop.data)
    if isempty(hofs)
        hof = HallOfFame(Vector())
        push!(pop.data, hof)
    else
        @assert length(hofs) == 1 "There should be exactly one Hall of Fame for the time being, found $(length(hofs))."
        hof = hofs[1]
    end
    # add inds to end of hall of fame
    for ind in pop.individuals
        push!(hof.individuals, ind)
    end
    # trim the hall of fame if it exceeds maxsize
    if length(hof.individuals) > maxsize
        hof.individuals = hof.individuals[end-maxsize+1:end]
    end
end

function sample_hall_of_fame_into_population!(state::AbstractState, pops::Vector{Vector{Population}}, pop_size::Int)
    if length(pops) != 1 || length(pops[1]) != 1
        @error "hall of fame not implemented for multiple populations yet"
    end
    pop = pops[1][1]
    hof = getonly(x-> x isa HallOfFame, pop.data)
    while length(pop.individuals) < pop_size
        if isempty(hof.individuals)
            @warn "Hall of Fame is empty, cannot sample into population $(pop.id)"
            break
        end
        # Sample an individual from the Hall of Fame without replacement
        idx = rand(state.rng, 1:length(hof.individuals))
        ind = hof.individuals[idx]
        # Check if the individual is already in the population
        if !any(i -> i === ind, pop.individuals)
            push!(pop.individuals, ind)
            deleteat!(hof.individuals, idx)
        end
    end
end
