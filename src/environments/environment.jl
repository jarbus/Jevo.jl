export step!, done, play
# Creation should be done as an environment constructor
function step!(env::AbstractEnvironment, args...; kwargs...)
    @error "step! not implemented for $(typeof(env))"
end
function done(::AbstractEnvironment)::Bool
    true
end
play(match::Match) = play(match.environment_creator, match.individuals)
function play(environment_creator::AbstractCreator, individuals::Vector{<:AbstractIndividual})
    play(environment_creator(), develop.(individuals))

end

function play(env::AbstractEnvironment, phenotypes::Vector{<:AbstractPhenotype})
    is_done = false
    scores = zeros(length(phenotypes))
    while !is_done
        scores += step!(env, phenotypes)
        is_done = done(env)
    end
    scores
end
