using Jevo
using CUDA
using Transformers
enable_gpu(CUDA.functional()) 
# 
Jevo.set_device()
@assert isdefined(Main, :jevo_device)
# check if gpu works

@assert Main.jevo_device([1]) isa CuArray "Worker unable to access GPU. You probably want to use a GPU on all workers."
