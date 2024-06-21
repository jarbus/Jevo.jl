using Jevo
using Logging
using StableRNGs

STATS_FILE = "statistics.h5"
isfile(STATS_FILE) && rm(STATS_FILE)

global_logger(JevoLogger())
rng = StableRNG(1)

k = 1
n_dims = 2
n_inds = 2
n_species = 2
n_gens = 10

counters = default_counters()
ng_gc = ng_genotype_creator = Creator(VectorGenotype, (n=n_dims,rng=rng))
ng_developer = Creator(VectorPhenotype)

comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, ng_gc, ng_developer) for i in 1:n_species], counters))
env_creator = Creator(CompareOnOne)

state = State(rng,[comp_pop_creator, env_creator],
    [InitializeAllPopulations(),
    AllVsAllMatchMaker(),
    Performer(),
    ScalarFitnessEvaluator(),
    TruncationSelector(k),
    CloneUniformReproducer(n_inds),
    Mutator(),
    PopSizeAssertor(n_inds),
    ClearInteractionsAndRecords(),
    create_op("Reporter",
        retriever=Jevo.get_individuals,
        operator=(s,is)-> foreach(i->println(i.genotype), is)
    ),

    create_op("Reporter", operator=(s,_)-> println(generation(s))
    ),
    Reporter(GenotypeSum, console=true)])

run!(state, n_gens)
