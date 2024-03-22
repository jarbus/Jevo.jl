using Jevo
using CUDA
using StableRNGs
using Logging
using Flux
using Transformers
using Transformers.TextEncoders

enable_gpu(CUDA.functional()) 
rng = StableRNG(1)
global_logger(JevoLogger())

rm("statistics.h5", force=true)

n_inds = 10
n_species = 1
n_tokens = 3
k = 1

startsym = "<s>"
endsym = "</s>"
unksym = "<unk>"
labels = string.(1:n_tokens)
vocab = [unksym, startsym, endsym, labels...]
vocab_size = length(vocab)

trf_args = (
  n_blocks = 2,
  n_heads = 2,
  head_dim = 5,
  hidden_dim = 10,
  ff_dim = 20,
  vocab_size = vocab_size,
 )

env_args = (
  vocab_size = vocab_size,
  batch_size = 100,
  seq_len = 2,
  n_repeat = 4,
)


textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)
phen_creator = Creator(TransformerPhenotype, (;textenc=textenc))

weight_cache = WeightCache(maxsize=10_000_000_000)
counters = default_counters()
gene_counter = find(:type, AbstractGene, counters)
geno_creator = Creator(Network, (rng, gene_counter, StrictCoupling, [
                          (Jevo.Transformer, trf_args),
                         ]))
# Composite pop
comp_pop_creator = Creator(CompositePopulation, ("species", [("p$i", n_inds, geno_creator, phen_creator) for i in 1:n_species], counters))
env_creator = Creator(RepeatSequence, env_args)

abstract type OutputValue <: AbstractMetric end
abstract type MatchCount <: AbstractMetric end

function evaluate(individual)
  model = develop(phen_creator, individual.genotype)
  infer(model, "1 2 1 2", print_output=true)
  Jevo.play(env_creator(), [model])[1]
end

BestReporter() = create_op("Reporter",
        retriever=Jevo.get_individuals,
        operator=(s,is)->
          @info StatisticalMeasurement(OutputValue,
                                       evaluate.(is),
                                       generation(s)))
state = State([comp_pop_creator, env_creator], 
      [InitializeAllPopulations(), 
        SoloMatchMaker(["p1"]), 
        Performer(),
        ScalarFitnessEvaluator(["p1"]), 
        TruncationSelector(k),
        BestReporter(),
        UniformReproducer(n_inds),
        Mutator(;mr=.01f0, n=2),
        PopSizeAssertor(n_inds),
        ClearInteractionsAndRecords(),
       ])
run!(state, 1)
