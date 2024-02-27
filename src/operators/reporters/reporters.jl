export Reporter

@define_op "Reporter" "AbstractReporter"
Reporter(type::Type{<:AbstractMetric}; h5=true, txt=true, console=false, kwargs...) =
    create_op("Reporter",
              operator=(s,_)->measure(type, s, h5, txt, console);kwargs...)

include("./timer.jl")
