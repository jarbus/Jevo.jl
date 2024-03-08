export MaxLogits
struct MaxLogits <: AbstractEnvironment
    n::Int
end

function step!(env::MaxLogits, models::Vector{Model})
    inputs = rand(Float32, env.n)
    @assert length(models) == 1 "Only one model is supported for now"
    models[1].chain(inputs)
end
(creator::Creator{MaxLogits})(;kwargs...) = MaxLogits(creator.kwargs...)
