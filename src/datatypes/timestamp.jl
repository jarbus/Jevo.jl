struct Timestamp <: AbstractData
    type::Type
    generation::Int
    start::DateTime
    stop::Union{DateTime, Nothing}
end
