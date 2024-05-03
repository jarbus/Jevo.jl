using Jevo
using CUDA
using Transformers
enable_gpu(CUDA.functional()) 
# check if gpu works
@assert todevice([1]) isa CuArray "Worker unable to access GPU. You probably want to use a GPU on all workers."



devs = collect(devices())
if myid() > 1
    @assert length(devs) == 1 "length(devs) == $(length(devs)) on $(myid())"
    dev = devs[1]
end
