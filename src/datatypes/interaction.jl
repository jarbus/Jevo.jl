struct Interaction <: AbstractInteraction
    match_id::Int
    individual_id::Int
    other_ids::Vector{Int}
    score::Float64
end
