export GenePooler
import Base: rand
@define_op "GenePooler" "AbstractOperator"
GenePooler(ids::Vector{String}=String[];n_back::Int,kwargs...) =
    create_op("AllVsAllMatchMaker",
          condition=always,
          retriever=Jevo.get_individuals,
          operator=(s,is)-> collect_genes(s,is,n_back),
          updater=override_genepool!;kwargs...)

function rand(rng::AbstractRNG, genepool::GenePool, dims::Tuple{Vararg{Int}})
    @assert haskey(genepool.genes, dims) "No genes found for dims $dims"
    rand(rng, genepool.genes[dims])
end


"""
Gets `n_back` genes from each individual in `inds` and stores them in the state's gene pool
"""
function collect_genes(state::AbstractState, inds::Vector{<:AbstractIndividual}, n_back::Int)
    genepool = GenePool()
    gene_counter = get_counter(AbstractGene, state)
    for ind in inds, weight in get_weights(state.rng, ind.genotype)
        n_genes = length(weight.muts)
        w_n_back = min(n_back, n_genes)
        dims = weight.dims
        for depth in 0:(w_n_back-1)
            gene = weight.muts[end-depth]
            # copy poolinfo if gene is from gene pool, else create new poolinfo
            poolinfo = gene.poolinfo.host_id > -1 ? gene.poolinfo : PoolInfo(ind.id, depth)
            genepool_gene = NetworkGene(inc!(gene_counter), gene.seed, gene.mr, gene.init!, poolinfo)
            if !haskey(genepool.genes, dims)
                genepool.genes[dims] = NetworkGene[genepool_gene]
            else
                push!(genepool.genes[dims], genepool_gene)
            end
        end
    end
    genepool
end

"""Goes through state.data, removes existing gene pool, and adds new gene pool"""
function override_genepool!(state::AbstractState, genepool::GenePool)
    filter!(x->!(x isa GenePool), state.data)
    push!(state.data, genepool)
end
