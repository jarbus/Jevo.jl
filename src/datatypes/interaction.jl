struct Interaction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    score::Float64
end

struct EstimatedInteraction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    score::Float64
end
