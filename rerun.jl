ENV["JULIA_STACKTRACE_ABBREVIATED"] = true
ENV["JULIA_STACKTRACE_MINIMAL"] = true
using Revise
using AbbreviatedStackTraces
# list all files in src directory recursively
files = vcat(split( read(`find src -type f -name "*.jl"`, String), "\n"),
             split( read(`find test -type f -name "*.jl"`, String), "\n"))
roc(files) do
    include("test/runtests.jl")
end
