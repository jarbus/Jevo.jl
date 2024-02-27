export TimeReporter

function log_time!(state::State, timestamps::Vector{Timestamp}, type::Type)
    if isempty(timestamps)
        push!(state.data, Timestamp(type, 0, now(), nothing))
    else
        @assert length(timestamps) == 1 "Expected only one timestamp of type $type, but found $(length(timestamps))"
        # log the time since the last timestamp
        elapsed_time = now() - timestamps[1].start 
        @info "Timer{$type}: $elapsed_time"
        # remove old timestamp from state.data
        filter!(t->typeof(t) <: TimeReporter && t.type != type, state.data)

    end
end
@define_op "TimeReporter" "AbstractReporter"

TimeReporter(type::Type; kwargs...) = create_op("TimeReporter",
            retriever=(s::AbstractState)->get_timestamps(s, type),
            updater=(s, ts)->log_time!(s, ts, type),
            kwargs...)
