# TODO: change init to svd/kaiming_normal where appropriate
NetworkGene(rng::AbstractRNG, counter::Counter, mr::Float16, init::Function=Jevo.kaiming_normal) = 
    NetworkGene(inc!(counter), rand(rng, UInt16), mr, init)

function Weights(rng::AbstractRNG, counter::AbstractCounter, dims::Tuple{Vararg{Int}})
    Weights(dims, [NetworkGene(rng, counter, Float16(1.0))])
end

function WeightCache(;maxsize::Int, by::Function=sizeof)
    LRU{Vector{NetworkGene}, Array{Float16}}(maxsize=maxsize, by=by)
end

function Network(rng::AbstractRNG, counter::AbstractCounter, coupling::Coupling, layers::Vector)
    """Create a network with a collection of layers and a coupling scheme"""
    for l in layers
        @assert length(l) == 3
        @assert l[1] <: AbstractLayer "Layer must be a subtype of AbstractLayer, got $(l[1])"
        @assert typeof(l[2]) <: Tuple{Vararg{Int}} "Dimensions must be a tuple of integers, got $(l[2])"
        @assert typeof(l[3]) <: Union{AbstractInitializer,Function} "Initializer must be a function or a subtype of AbstractInitializer, got $(l[3])"
    end
    Network(coupling, [l[1](l[2], l[3], counter, rng) for l in layers])
end

function Dense(dims::Tuple{Vararg{Int}}, σ::Function, counter::AbstractCounter, rng::AbstractRNG)
    """Create a dense layer with a weight matrix and a bias vector"""
    @assert length(dims) == 2 "Dense layer must have 2 dimensions, got $(length(dims))"
    weights = Weights(rng, counter, (dims[2], dims[1]))
    bias = Weights(rng, counter, (dims[2],))
    Dense(weights, bias, σ)
end
