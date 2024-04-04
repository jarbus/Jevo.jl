"""
If initialized with ones or zeros, then just mutate with Gaussian noise
"""
function compute_init(fn::Function)
    if fn == apply_one! || fn == apply_zero!
        return apply_gaussian_normal_noise!
    end
    fn
end

function find_highest_mr(weights::Vector{Weights}, lookback::Int)
    @assert lookback >= 0 "Lookback must be non-negative"
    max_mr = -1
    for weight in weights, gene in weight.muts[end-min(length(weight.muts),lookback)+1:end]
        max_mr = max(max_mr, gene.mr)
    end
    @assert max_mr >= 0 "max_mr must be non-negative"
    max_mr
end
compute_max_mr(::Vector{Weights}, mr::Float32, ::Int) = mr
# NOTE that this automatically sets the max mutation rate to the highest mutation rate for the first 20 muts or so
# since initialization is an MR of 1
compute_max_mr(weights::Vector{Weights}, mr::Tuple{Vararg{Float32}}, lookback::Int) =
    lookback < 0 ? rand(mr) : find_highest_mr(weights, lookback)
sample_mr(::AbstractRNG, mr::Float32) = mr
sample_mr(rng::AbstractRNG, mr::Tuple{Vararg{Float32}}) = rand(rng, mr)

function mutate_weights!(rng::AbstractRNG, state::State, genotype::Network, mr::Union{Float32,Tuple{Vararg{Float32}}}, lookback::Int; n=-1)
    gene_counter = get_counter(AbstractGene, state)
    weights = get_weights(rng, genotype, n=n)
    # Compute a maximum mutation rate, don't add genes
    # with mutation rates higher than this
    # This guarantees that we generate some mutations with small
    # updates and some with large updates. If we sampled uniformly,
    # we would generate a lot of mutations with at least one 
    # large update
    max_mr = compute_max_mr(weights, mr, lookback)
    for weight in weights
        mrf0 = sample_mr(rng, mr)
        mrf0 > max_mr && continue
        init = compute_init(weight.muts[1].init!)
        if rand(rng) > 0.05 # Generate new gene with 95% probability
            push!(weight.muts, NetworkGene(rng, gene_counter, mrf0, init))
        else # Re-apply existing gene with 5% probability
            push!(weight.muts, NetworkGene(gene_counter, weight.muts[end].seed, mrf0, init))
        end
    end
end

function mutate(rng::AbstractRNG, state::State, genotype::Network; mr::Union{Float32,Tuple{Vararg{Float32}}}, n::Int=2, lookback::Int=-1)
    @assert genotype.coupling in (StrictCoupling, NoCoupling) "Only strict and no coupling are supported right now"
    new_genotype = deepcopy(genotype)
    genotype.coupling == StrictCoupling && (n = -1)
    mutate_weights!(rng, state, new_genotype, mr, lookback, n=n)
    new_genotype
end


