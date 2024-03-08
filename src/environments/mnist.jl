export MNISTEnv

mnist_trn = MNIST(split = :train)[:]
const trnX = Float32.(reshape(mnist_trn.features, :, size(mnist_trn.features, 3)))
# convert to one-hot
@assert minimum(mnist_trn.targets) == 0
@assert maximum(mnist_trn.targets) == 9
const trnY = OneHotArrays.onehotbatch(mnist_trn.targets, 0:9)
@assert size(trnX, 1) == 784
@assert size(trnX, 2) == 60000
@assert minimum(trnY) == 0
@assert maximum(trnY) == 1

struct MNISTEnv <: AbstractEnvironment end

cross_entropy_loss(outputs, trnY) = -sum(trnY .* log.(outputs)) / size(trnY, 2)


function step!(::MNISTEnv, phenotypes::Vector{VectorPhenotype})
    @assert length(phenotypes) == 1
    m = phenotypes[1].chain
    outputs = m(trnX) # 10x60000
    # cross entropy loss between outputs and trnY
    loss = cross_entropy_loss(outputs, trnY)


    
end
(creator::Creator{MNISTEnv})(;kwargs...) = MNISTEnv()
