export Dense, NetworkGene, Network, Weights, StrictCoupling, LooseCoupling, NoCoupling, WeightCache, GenotypeCache, AbstractModel
abstract type AbstractInitializer <: Function end
abstract type AbstractWeights end
abstract type AbstractLayer <: AbstractGenotype end
abstract type AbstractMutation end
abstract type AbstractModel end

"""
    @enum Coupling Strict Loose None

# Values
- `StrictCoupling`: Weights in all sub-layers are mutated using the same initial rng seed
- `LooseCoupling`: Weights in all sub-layers can be mutated independently or together
- `NoCoupling`: Weights in all sub-layers are mutated independently
"""
@enum Coupling StrictCoupling LooseCoupling NoCoupling

# Weights
struct NetworkGene <: AbstractMutation
    id::Int
    seed::UInt16
    mr::Float16
    init::Union{AbstractInitializer,Function}
end

struct Weights{N} <: AbstractWeights where N <: Int
    dims::NTuple{N, Int}
    muts::Vector{NetworkGene}
end

struct WeightsCollection <: AbstractWeights
    """Concatenation of multiple weight blocks into a single weight tensor, to adjust subsets of weights independently"""
    weights::Array{AbstractWeights}
end

struct FactorWeight <: AbstractWeights
    """Low-rank factorization of a weight matrix"""
    A::AbstractWeights
    B::AbstractWeights
end

struct Network <: AbstractLayer
    """A collection and a coupling scheme."""
    coupling::Coupling
    layers::Vector{<:AbstractLayer}
end

@define_op "NetworkGeneMutator" "AbstractMutator"
NetworkGeneMutator(ids::Vector{String}=String[]; mr::Float16=Float16(0.001), kwargs...) = 
    create_op("NetworkGeneMutator", 
              retriever=PopulationRetriever(ids),
              updater=map(map((s,p)->mutate!(s, p, mr=mr)));
              kwargs...)

_WeightCache = Union{LRU{Vector{NetworkGene}, <:Array{Float16}}, Nothing}
# so we only need to transmit delta genotypes
GenotypeCache = Union{LRU{Int, Network}, Nothing}

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

############################
# PERFORMANCE CRITICAL START  (suspected)
# possible optimizations: @inbounds, @fastmath
function get_earliest_cached_weight(dims::Tuple{Vararg{Int}}, genes::Vector{NetworkGene}, weight_cache::_WeightCache)
    """Return the earliest cached weight in the gene list. If none are cached, return a zero tensor of the given dimensions. Allocates memory. Also returns the idx of the earliest cached gene."""
    arr = zeros(Float16, dims)
    weight_cache === nothing && return arr, 0
    for i in length(genes):-1:1
        if genes[1:i] in keys(weight_cache)
            arr += weight_cache[genes[1:i]]
            return arr, i
        end
    end
    arr, 0
end
function tensor(w::Weights; weight_cache::_WeightCache=nothing)
    # ADD MUTS that have have not been cached
    dims, genes = w.dims, w.muts
    # get earliest cached weight or zero tensor if none found
    arr, ancestor_idx = get_earliest_cached_weight(dims, genes, weight_cache)
    yes_weight_cache = !isnothing(weight_cache)
    # iteratively apply remaining mutations
    for i in ancestor_idx+1:length(genes)
        gene = genes[i]
        arr .+= gene.mr .* gene.init(StableRNG(gene.seed), Float16, dims...)
        # update cache if we are using one
        yes_weight_cache && (weight_cache[genes[1:i]] = copy(arr))
    end
    arr
end
# PERFORMANCE CRITICAL END
############################

function develop(::Creator, network::Network)
    # get global variable Main.weight_cache for weight cache
    # check if weight_cache is defined
    if !isdefined(Main, :weight_cache)
        @warn "No weight cache found, looking specifically for a variable called `weight_cache` in the global scope. Main has the following variables: $(names(Main))"
        Main.weight_cache = nothing
    end

    weight_cache = Main.weight_cache
    Chain(create_layer.(network.layers, weight_cache=weight_cache)...)
end

function create_weight!(weight_cache::LRU, w::AbstractWeights)
    """Converts the genotype of a weight matrix
    (RNGs, MRs, factors) to the phenotype."""
end

function mutate(state::State, genotype::Network)
    rng = state.rng
    genotype
end

struct SVDInitializer <: AbstractInitializer
    """Initialize a weight matrix by sampling rows/columns from a singular value decomposition of an existing initializer `fn`.
    To be used as part of a mutation, so it does not need to hold an rng, mr, or dims"""
    fn::Function
end

struct Dense <: AbstractLayer
    weights::AbstractWeights
    bias::AbstractWeights
    σ::Any
end

function Dense(dims::Tuple{Vararg{Int}}, σ::Function, counter::AbstractCounter, rng::AbstractRNG)
    """Create a dense layer with a weight matrix and a bias vector"""
    @assert length(dims) == 2 "Dense layer must have 2 dimensions, got $(length(dims))"
    weights = Weights(rng, counter, (dims[2], dims[1]))
    bias = Weights(rng, counter, (dims[2],))
    Dense(weights, bias, σ)
end

function create_layer(layer::Dense; weight_cache::_WeightCache)
    weights = tensor(layer.weights, weight_cache=weight_cache)
    bias = tensor(layer.bias, weight_cache=weight_cache)
    Flux.Dense(weights, bias, layer.σ)
end

struct NetworkInstantiator
    weight_cache::_WeightCache
    genotype_cache::GenotypeCache
end

function get_nearest_ancestor(ancestors::Vector{Int}) 
    """Return nearest cached ancestor from list"""
end
function send_delta_genotype(ancestor::Int, descendant::Int, genotype::Network)
    """Send diff of nearest cached ancestor and descendant and caches the result"""
end
