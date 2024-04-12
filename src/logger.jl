export JevoLogger, @h5

const H5_LOG_LEVEL = LogLevel(5)
const datefmt = "yy-mm-dd HH:MM:SS"
import Base: log
# log to the HDF5 file and defaults to the HDF5_LOG_LEVEL
macro h5(exs...) Base.CoreLogging.logmsg_code((Base.CoreLogging.@_sourceinfo())..., esc(H5_LOG_LEVEL), exs...) end

struct HDF5Logger <: AbstractLogger
    path::AbstractString
    io::HDF5.File
    always_flush::Bool
end

function HDF5Logger(path::String;kwargs...)
    io = h5open(path, "cw")
    HDF5Logger(path, io, true)
end

function JevoLogger(;hdf5_path::String="statistics.h5", log_path::String="run.log")
    filelogger = FormatLogger(log_path) do io, args
        println(io, "$(Dates.format(now(), datefmt)) $(args.message)")
    end
    MinLevelLogger(
        TeeLogger(
            EarlyFilteredLogger(log->log.level == H5_LOG_LEVEL, HDF5Logger(hdf5_path)),
            EarlyFilteredLogger(log->log.level ∈ (Info, Warn, Error) , filelogger)),
        Info)
end


function Base.CoreLogging.handle_message(logger::HDF5Logger, level, m::XPlot.AbstractMeasurement, _module, group, id, file, line; kwargs...)
    @assert level.level == H5_LOG_LEVEL.level
    @assert length(kwargs) == 0
    pidpath = logger.path*".pid"
    monitor = FileWatching.Pidfile.mkpidlock(pidpath, wait=true)
    fullpath = joinpath(string(m.iteration), string(m.metric))
    for field in fieldnames(typeof(m))
        (field == :metric || field == :iteration) && continue
        logger.io[joinpath([fullpath, string(field)])] = getfield(m, field)
    end
    flush(logger.io)
    close(monitor)
end

function Base.log(m::XPlot.AbstractMeasurement, h5::Bool, txt::Bool, console::Bool)
    h5 && @h5 m
    txt && @info m
    console && println(m)
end

Base.CoreLogging.shouldlog(logger::HDF5Logger, arg...) = true
Base.CoreLogging.min_enabled_level(::HDF5Logger) = BelowMinLevel
Base.CoreLogging.catch_exceptions(logger::HDF5Logger) = Base.CoreLogging.catch_exceptions(logger)
