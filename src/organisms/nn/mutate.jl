"""
If initialized with ones or zeros, then just mutate with Gaussian noise
"""
function compute_init(fn::Function)
    if fn == apply_one! || fn == apply_zero!
        return apply_gaussian_normal_noise!
    end
    fn
end

function mutate_weights!(rng::AbstractRNG, state::State, genotype::Network, mr::Union{Float32,Tuple{Vararg{Float32}}}; n=-1)
    gene_counter = get_counter(AbstractGene, state)
    weights = get_weights(rng, genotype, n=n)
    for weight in weights
        mrf0 = mr isa Float32 ? mr : rand(rng, mr)
        init = compute_init(weight.muts[1].init!)
        if rand(rng) > 0.05 # Generate new gene with 95% probability
            push!(weight.muts, NetworkGene(rng, gene_counter, mrf0, init))
        else # Re-apply existing gene with 5% probability
            push!(weight.muts, NetworkGene(gene_counter, weight.muts[end].seed, mrf0, init))
        end
    end
end

function mutate(rng::AbstractRNG, state::State, genotype::Network; mr::Union{Float32,Tuple{Vararg{Float32}}}, n::Int=2)
    @assert genotype.coupling in (StrictCoupling, NoCoupling) "Only strict and no coupling are supported right now"
    new_genotype = deepcopy(genotype)
    genotype.coupling == StrictCoupling && (n = -1)
    mutate_weights!(rng, state, new_genotype, mr, n=n)
    new_genotype
end


