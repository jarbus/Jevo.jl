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
    n_dims = 1
    n_inds = 2
    n_species = 2
    n_pops = 2
    counters = default_counters()
    ng_gc = ng_genotype_creator = Creator(VectorGenotype, (n=n_dims,rng=rng))
    ng_phenotype_creator = Creator(VectorPhenotype)
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
                            [CompositePopulation("composite$i",
                                [("p$i", n_inds, ng_gc) for i in 1:n_species],
                            counters)
                        for i in 1:n_pops])
        # Create CompositePopulation creator
        """
            ecosystem
            ├── composite1
            │   ├── pop1: ind1a1, ind1a2
            │   └── pop2: ind1b1, ind1b2
            └── composite2
                ├── pop3: ind2a1, ind2a2
                └── pop4: ind2b1, ind2b2
        """
        pid = 0
        comp_comp_pop_creator = 
            Creator(CompositePopulation, 
                ("ecosystem", 
                 [Creator(CompositePopulation,
                        ("composite$i",
                         [("p$(pid+=1)", n_inds, ng_gc) for p in 1:n_species],
                        counters))
                for i in 1:n_pops]))
        comp_comp_pop = comp_comp_pop_creator()
        pop_initializer = InitializeAllPopulations()
        state = State([comp_comp_pop_creator], [pop_initializer])
        @testset "PopulationCreatorRetriever" begin
            pops = [pop_creator() for pop_creator in pop_initializer.retriever(state)]
            @test pops |> length == 1
            @test pops[1].id == "ecosystem"
            run!(state, 1)
        end
        @testset "PopulationRetriever" begin
            @test state.populations |> length == 1
            @test state.populations[1].id == "ecosystem"
            @test state.populations[1].populations |> length == n_pops
            pops = PopulationRetriever(["ecosystem"])(state)
            @test pops |> length == 1 # requesting ecosystem returns single population
            @test length(pops[1]) == 8
            pops = PopulationRetriever(["nopop"])(state)
            @test pops |> length == 1
            @test length(pops[1]) == 0
            println("composite1")
            pops = PopulationRetriever(["composite1"])(state)
            @test length(pops) == 1
            @test length(pops[1]) == 4
            pops = PopulationRetriever(["p2"])(state)
            @test length(pops) == 1
            @test length.(pops) == [2]
            pops = PopulationRetriever(["p1", "p2"])(state)
            @test length(pops) == 2
            @test length.(pops) == [2,2]
        end

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
