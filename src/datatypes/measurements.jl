export Measurement, StatisticalMeasurement, measure
struct Measurement <: AbstractMeasurement
    metric::Union{String,Type{<:AbstractMetric}}
    value::Any
    generation::Int
end
struct StatisticalMeasurement <: AbstractMeasurement
    metric::Union{String, Type{<:AbstractMetric}}
    min::Float64
    mean::Float64
    std::Float64
    max::Float64
    n_samples::Int
    generation::Int
end

function StatisticalMeasurement(type::Union{String, Type{<:AbstractMetric}}, data::Vector{<:Real}, generation::Int)
    StatisticalMeasurement(type, minimum(data), mean(data), std(data), maximum(data), length(data), generation)
end

measure(metric::AbstractMetric, ::AbstractState) = @error "No measure function defined for $(typeof(metric))"

Base.show(io::IO, m::Measurement; digits::Int=3) = print(io, "gen=$(m.generation) $(m.metric)=$(round(m.value, digits=digits))")
Base.show(io::IO, m::StatisticalMeasurement; digits::Int=3) = print(io, 
    "gen=$(m.generation) $(m.metric): |$(round(m.min, digits=digits)), $(round(m.mean, digits=digits)) ± $(round(m.std, digits=digits)), $(round(m.max, digits=digits))|, $(m.n_samples) samples")


HEAD="iter"
write(f, m::Measurement) = f[joinpath(HEAD,"$(m.generation)/$(m.metric)")] = m.value
function write(f, m::StatisticalMeasurement)
    head = joinpath(HEAD, "$(m.generation)/$(m.metric)")
    f[joinpath(head, "min")] = m.min
    f[joinpath(head, "mean")] = m.mean
    f[joinpath(head, "std")] = m.std
    f[joinpath(head, "max")] = m.max
    f[joinpath(head, "n_samples")] = m.n_samples
end

