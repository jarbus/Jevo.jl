function mutate_weights!(rng::AbstractRNG, state::State, genotype::Network, mr::Float32; n=-1)
    gene_counter = get_counter(AbstractGene, state)
    weights = get_weights(rng, genotype, n=n)
    for weight in weights
        init = weight.muts[1].init!
        if rand(rng) > 0.05 # Generate new gene with 95% probability
            push!(weight.muts, NetworkGene(rng, gene_counter, mr, init))
        else # Re-apply existing gene with 5% probability
            push!(weight.muts, NetworkGene(gene_counter, weight.muts[end].seed, mr, init))
        end
    end
end

function mutate(rng::AbstractRNG, state::State, genotype::Network; mr::Union{Float32,Tuple{Vararg{Float32}}}, n::Int=2)
    @assert genotype.coupling in (StrictCoupling, NoCoupling) "Only strict and no coupling are supported right now"
    new_genotype = deepcopy(genotype)
    genotype.coupling == StrictCoupling && (n = -1)
    mrf0 = mr isa Float32 ? mr : rand(rng, mr)
    mutate_weights!(rng, state, new_genotype, mrf0, n=n)
    new_genotype
end


