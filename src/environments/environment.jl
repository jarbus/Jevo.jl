export step!, done, play
# Creation should be done as an environment constructor
done(::AbstractEnvironment)::Bool = true
play(match::Match) = play(match.environment_creator, match.individuals)

function play(c::Creator{E}, inds::Vector{I}) where {E<:AbstractEnvironment, I<:AbstractIndividual}
    play(c(), develop(inds))
end

function play(env::E, phenotypes::Vector{P}) where {E <: AbstractEnvironment, P<:AbstractPhenotype}
    is_done = false
    scores = zeros(Float32, length(phenotypes))
    while !is_done
        scores .+= step!(env, phenotypes)
        is_done = done(env)
    end
    @assert all(-Inf < score < Inf for score in scores)
    scores
end
