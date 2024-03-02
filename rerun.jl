println("==================================")
ENV["JULIA_STACKTRACE_ABBREVIATED"] = true
ENV["JULIA_STACKTRACE_MINIMAL"] = true
ENV["JULIA_TEST_MODE"] = "true"
using AbbreviatedStackTraces
include("test/runtests.jl")
