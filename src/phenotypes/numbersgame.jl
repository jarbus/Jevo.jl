export VectorPhenotype
mutable struct VectorPhenotype <: AbstractPhenotype
    numbers::Vector{Float32}
end

develop(c::Creator, genotype::VectorGenotype) = c.type(genotype.numbers)
