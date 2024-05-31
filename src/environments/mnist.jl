# export MNISTEnv
#
# const N_SAMPLES = 60_000
# const BATCH_SIZE = 10_000
#
# function initialize_data()
#     mnist_trn = MNIST(split = :train)[:]
#     trnX = reshape(mnist_trn.features, :, size(mnist_trn.features, 3))[:,1:N_SAMPLES] |> gpu
#     trnY = OneHotArrays.onehotbatch(mnist_trn.targets, 0:9)[:,1:N_SAMPLES] |> gpu
#     return trnX, trnY
# end
#
# struct MNISTEnv <: AbstractEnvironment end
#
#
# function step!(::MNISTEnv, phenotypes::Vector{Model})
#     if !isdefined(Main, :trnX)
#          Main.trnX, Main.trnY = initialize_data()
#     end
#     trnX, trnY = Main.trnX, Main.trnY
#     @assert length(phenotypes) == 1
#     m = phenotypes[1].chain |> gpu
#     # iterates over training data in 10_000 sample batches
#     losses = Float32[]
#     for i in 1:BATCH_SIZE:size(trnX, 2)
#         inputs = trnX[:,i:i+BATCH_SIZE-1]
#         outputs = m(inputs) # 10x60000
#         neg_loss = -Flux.logitcrossentropy(outputs, trnY[:,i:i+BATCH_SIZE-1]) |> cpu
#         @assert -Inf < neg_loss < Inf
#         push!(losses, neg_loss)
#     end
#     mean_loss = mean(losses)
#     Float32[mean_loss]
# end
#
# (creator::Creator{MNISTEnv})(;kwargs...) = MNISTEnv()
