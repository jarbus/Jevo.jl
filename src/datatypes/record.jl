# Generic record, typically used for fitness-proportional selection
struct Record <: AbstractRecord
    id::Int
    fitness::Float64
end
