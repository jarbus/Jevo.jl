start_time = time()
using Jevo
using HDF5
using Test
using StableRNGs
using Logging
using StatsBase
using Flux
using Transformers
using Transformers.TextEncoders
using Transformers.Datasets: batched
using PhylogeneticTrees
using CUDA
using Distributed
#Jevo.set_device()

# Global variable for weight_cache
weight_cache = WeightCache(maxsize=1_000_000)

rng = StableRNG(1)

#= @testset "counter" begin =#
#=   """Creates a counter, counts 10 times, and checks that the final value is correct.""" =#
#=   c = Counter(AbstractGene) =#
#=   for i in 1:10 =#
#=       inc!(c) =#
#=   end =#
#=   @test value(c) == 11 =#
#=   @test find(:type, AbstractGene, [c]) == c =#
#=   try =#
#=       find(:type, AbstractIndividual, [c]) =#
#=   catch =#
#=       @test true =#
#=   end =#
#= end =#
#==#
#= @testset "state" begin =#
#=   state = State() =#
#=   Jevo.operate!(state) =#
#=   @test Jevo.get_counter(AbstractGeneration, state) |> value == 2 =#
#= end =#

#include("./test-ng-integration.jl")
#include("./test-writers.jl")
#include("./test-phylo.jl")
#include("./test-nn.jl")
#include("./test-text-environments.jl")
#include("./test-traverse.jl")
#include("./test-trade.jl")
#include("./test-phylo.jl")
#include("./test-trade.jl")
include("./test-clustering.jl")
include("./test-nsgaii.jl")
end_time = time()
println("Tests passed in $(end_time - start_time) seconds.")
