export TimeReporter
struct TimeReporter <: AbstractReporter
    condition::Function
    retriever::Function
    operator::Function
    updater::Function
end
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
function TimeReporter(type::Type)
    TimeReporter(always,
            (s::AbstractState)->get_timestamps(s, type),
            noop,
            (s, ts)->log_time!(s, ts, type)
           )
end
