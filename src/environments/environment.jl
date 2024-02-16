# Creation should be done as an environment constructor
function step!(::AbstractEnvironment, args...; kwargs...)
end
function done(::AbstractEnvironment)::Bool
    true
end
function play(environment_creator::AbstractCreator, genotypes_and_creators::Vector{Tuple{<:AbstractGenotype, <:AbstractCreator}})
    play(environment_creator(), [develop(c, g) for (g, c) in genotypes_and_creators])

end

function play(env::AbstractEnvironment, phenotypes::Vector{<:AbstractPhenotype})
    is_done = false
    scores = zeros(length(phenotypes))
    while !is_done
        scores += step!(env)
        is_done = done(env)
    end
    scores
end
