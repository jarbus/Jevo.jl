using Jevo
using CUDA
using Transformers
using Flux
enable_gpu(CUDA.functional()) 
# 
Jevo.set_device()
@assert isdefined(Main, :jevo_device_id)
# check if gpu works
device!(Main.jevo_device_id)
gpu_test = gpu(rand(1))
@assert gpu_test isa CuArray
