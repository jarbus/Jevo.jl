# Generic record, typically used for fitness-proportional selection
struct Record <: AbstractRecord
    id::Int
    fitness::Float64
end

struct LexicaseRecord <: AbstractRecord
    id::Int
    fitness::Dict{Int, Float32}
end
