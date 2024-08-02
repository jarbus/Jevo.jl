using Jevo
using StatsBase
using Random
using StableRNGs
using CUDA
using Transformers
using Flux
enable_gpu(CUDA.functional()) 
# 
Jevo.set_device()
@assert isdefined(Main, :jevo_device_id)
# check if gpu works
device!(Main.jevo_device_id)
gpu_test = gpu([1.0])
@assert gpu_test isa CuArray
