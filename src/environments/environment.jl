export step!, done, play
# Creation should be done as an environment constructor
done(::AbstractEnvironment)::Bool = true
play(match::Match) = play(match.environment_creator, match.individuals)
step!(env::AbstractEnvironment, args...; kwargs...) =
    @error "step! not implemented for $(typeof(env))"

play(environment_creator::AbstractCreator, individuals::Vector{<:AbstractIndividual}) =
    play(environment_creator(), develop.(individuals))

function play(env::AbstractEnvironment, phenotypes::Vector{<:AbstractPhenotype})
    is_done = false
    scores = zeros(Float64, length(phenotypes))
    while !is_done
        scores .+= step!(env, phenotypes)
        is_done = done(env)
    end
    scores
end
