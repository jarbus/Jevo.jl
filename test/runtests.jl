ENV["JULIA_STACKTRACE_ABBREVIATED"] = true
ENV["JULIA_STACKTRACE_MINIMAL"] = true
using Jevo
using Test
using AbbreviatedStackTraces
using StableRNGs

@testset "Jevo.jl" begin

rng = StableRNG(1)

@testset "counter" begin
    """Creates a counter, counts 10 times, and checks that the final value is correct."""
    c = Counter(AbstractGene)
    for i in 1:10
        inc!(c)
    end
    @test value(c) == 11
    @test find(:type, AbstractGene, [c]) == c
    try
        find(:type, AbstractIndividual, [c])
    catch
        @test true
    end

end

@testset "numbers game" begin
    # creators
    n_dims = 3
    n_inds = 10
    n_species = 2
    n_pops = 2
    counters = default_counters()
    ng_gc = ng_genotype_creator = Creator(VectorGenotype, (n=n_dims,rng=rng))
    ng_phenotype_creator = Creator(VectorPhenotype)
    # ng_ic = Creator(Individual, counter, )
    @testset "creation" begin
        # genotypes
        genotype = ng_genotype_creator()
        @test length(genotype.numbers) == n_dims
        # phenotypes
        phenotype = develop(ng_phenotype_creator, genotype)
        @test phenotype.numbers == genotype.numbers
        # Test that Individual has randomly generated data
        @test Individual(counters, ng_genotype_creator).genotype.numbers |> sum != 0
        # Population
        @test Population("ng", n_inds, ng_genotype_creator, counters).population |> length == n_inds
        # Composite population
        comp_pop = CompositePopulation("species", [("$i", n_inds, ng_gc) for i in 1:n_species], counters)
        # Finally, create a composite population creator
        comp_pop_creator = Creator(CompositePopulation, ("species", [("$i", n_inds, ng_gc) for i in 1:n_species], counters))
        comp_pop = comp_pop_creator()
        @test comp_pop.populations |> length == n_pops
        @test all(length(p.population) == n_inds for p in comp_pop.populations)


        @testset "environment" begin
        end

        @testset "interaction" begin
        end

        @testset "match" begin
        end
    end

    @testset "operators" begin
        @testset "matchmaker" begin
        end

        @testset "evaluators" begin
        end

        @testset "scorers" begin
        end

        @testset "replacer" begin
        end

        @testset "selectors" begin
        end

        @testset "mutators" begin
        end

        @testset "reporter" begin
        end

        @testset "assertor" begin
        end
    end
end

@testset "neural-net" begin
end
end
