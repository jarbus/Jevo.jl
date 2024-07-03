println("\n"^30)
ENV["JULIA_STACKTRACE_ABBREVIATED"] = true
ENV["JULIA_STACKTRACE_MINIMAL"] = true
using AbbreviatedStackTraces
include("test/runtests.jl")
