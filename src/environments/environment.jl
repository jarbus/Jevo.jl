export step!, done, play
using PythonCall
# Creation should be done as an environment constructor
done(::AbstractEnvironment)::Bool = true
play(match::Match) = play(match.environment_creator, match.individuals)


function play(c::Creator{E}, inds::Vector{I}) where {E<:AbstractEnvironment, I<:AbstractIndividual}
    isdefined(Jevo, :jevo_device_id) &&  device!(Jevo.jevo_device_id)
    lock(Jevo.get_env_lock()) do
        phenotypes = develop.(inds) .|> gpu
        ids = [ind.id for ind in inds]
        play(c(), ids, phenotypes)
    end
end

function play(env::E, ids::Vector{Int}, phenotypes::Vector{P}) where {E <: AbstractEnvironment, P<:AbstractPhenotype}
    is_done = false
    interactions = Vector{Interaction}()
    while !is_done
        new_interactions = step!(env, ids, phenotypes)
        append!(interactions, new_interactions)
        is_done = done(env)
    end
    @assert length(interactions) > 0
    @assert any(i->!isinf(i.score), interactions) "All interactions returned an infinite score"
    interactions
end
