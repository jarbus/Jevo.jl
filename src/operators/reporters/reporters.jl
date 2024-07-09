export Reporter

"""
    Reporter(;kwargs...)

    Reporter(type::Type{<:AbstractMetric}; h5=true, txt=true, console=false, kwargs...)

Operator that "reports" some aspect of the state to either the console, a text file, or an hdf5 file. Allows us to log or record data anywhere in the pipeline.
"""
@define_op "Reporter" "AbstractReporter"
Reporter(type::Type{<:AbstractMetric}; h5=true, txt=true, console=false, kwargs...) =
    create_op("Reporter",
              operator=(s,_)->measure(type, s, h5, txt, console);kwargs...)
