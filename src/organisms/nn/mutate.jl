function mutate_weights!(rng::AbstractRNG, state::State, genotype::Network, mr::Float32; n=-1)
    gene_counter = get_counter(AbstractGene, state)
    genepool = nothing
    try
        genepool = findone(GenePool, state.data)
    catch
        @warn("Gene pool not found in state for mutation, should only occur in testing")
        genepool = nothing
    end
    weights = get_weights(rng, genotype, n=n)
    for weight in weights
        init = weight.muts[1].init!
        if isnothing(genepool) || rand(rng) > 0.1 # Generate new gene with 90% probability
            push!(weight.muts, NetworkGene(rng, gene_counter, mr, init))
        else # Re-apply existing gene with 10% probability
            gp_gene = rand(rng, genepool, weight.dims)
            push!(weight.muts, NetworkGene(gene_counter, gp_gene.seed, mr, init, gp_gene.poolinfo))
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


