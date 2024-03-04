export MaxLogits
struct MaxLogits <: AbstractEnvironment end

function step!(::MaxLogits, models::Vector{Model})
    @assert length(models) == 2
    inputs = ones(Float16, 5)
    outputs = Vector{Float16}(undef, 2)
    for (i, m) in enumerate(models)
        output = m.chain(inputs)
        @assert length(output) == 1
        outputs[i] = output[1]
    end
    outputs[1] == outputs[2] && return [0.5, 0.5]
    scores = zeros(Float16, 2)
    scores[argmax(outputs)] = 1.0
    scores
end
(creator::Creator{MaxLogits})(;kwargs...) = MaxLogits()
