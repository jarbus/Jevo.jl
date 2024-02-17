println("==================================")
ENV["JULIA_STACKTRACE_ABBREVIATED"] = true
ENV["JULIA_STACKTRACE_MINIMAL"] = true
ENV["JULIA_TEST_MODE"] = "true"
using AbbreviatedStackTraces
using Jevo
using Test
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

@testset "state" begin
    state = State()
end

@testset "numbers game" begin
    # creators
    n_dims = 3
    n_inds = 2
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
        @test Population("ng", n_inds, ng_genotype_creator, counters).individuals |> length == n_inds
        # Composite population
        comp_pop = CompositePopulation("species", [("p$i", n_inds, ng_gc) for i in 1:n_species], counters)
        # create a composite population creator
        comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_gc) for i in 1:n_species], counters))
        comp_pop = comp_pop_creator()
        @test comp_pop.populations |> length == n_pops
        @test all(length(p.individuals) == n_inds for p in comp_pop.populations)
        # Create composite composite population, 2 deep binary tree
        comp_comp_pop = CompositePopulation("ecosystem",
                            [CompositePopulation("species",
                                [("p$i", n_inds, ng_gc) for i in 1:n_species],
                            counters)
                        for i in 1:n_pops])
        # Create CompositePopulation creator
        """
            ecosystem
            ├── composite1
            │   ├── pop1a: ind1a1, ind1a2
            │   └── pop1b: ind1b1, ind1b2
            └── composite2
                ├── pop2a: ind2a1, ind2a2
                └── pop2b: ind2b1, ind2b2
        """
        comp_comp_pop_creator = 
            Creator(CompositePopulation, 
                ("ecosystem", 
                 [Creator(CompositePopulation,
                        ("species",
                        [("p$i", n_inds, ng_gc) for i in 1:n_species],
                        counters))
                for i in 1:n_pops]))
        comp_comp_pop = comp_comp_pop_creator()
        state = State([comp_comp_pop_creator],
                      [InitializeAllPopulations()])
        run!(state, 1)
        

        # PopulationRetriever 
        """
        ids = ["ecosystem"] or [] will fetch:
            [[inds1a1, ind1a2, ind1b1, ind1b2],
             [ind2a1, ind2a2, ind2b1, ind2b2]]
            which flattens each subpopulations into a single vector
            and returns a vector of these vectors

        ids = ["composite1"] or ["pop1a", "pop1b"] will fetch:
            [[ind1a1, ind1a2], [ind1b1, ind1b2]]

        ids = ["pop1a"] will fetch:
            [[ind1a1, ind1a2]]
        """
        pop_retriever = PopulationRetriever(["species"])


        # Env
        env_creator = Creator(CompareOnOne)
        env = env_creator()

        @testset "match" begin
        end


        @testset "interaction" begin
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
