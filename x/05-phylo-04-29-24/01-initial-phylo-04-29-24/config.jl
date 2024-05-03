using Distributed
@everywhere begin
  using CUDA
  using Transformers
  enable_gpu(CUDA.functional()) 
  using Jevo
  using StableRNGs
end
using Flux
using Transformers.TextEncoders
using Logging

rng = StableRNG(2)

n_weights_to_mutate = -1
lookback = 20
n_inds = 1000
n_species = 1
n_tokens = 9
k = 1
n_generations = 10


startsym = "<s>"
endsym = "</s>"
unksym = "<unk>"
labels = string.(1:n_tokens)
vocab = [unksym, startsym, endsym, labels...]
vocab_size = length(vocab)
checkpointname = "check.jls"

trf_args = (
  n_blocks = 2,
  n_heads = 8,
  head_dim = 32,
  hidden_dim = 128,
  ff_dim = 256,
  qkv_rank=8,
  o_rank=8,
  ff_rank=32,
  embed_rank=-1,
  vocab_size = vocab_size,
 )

env_args = (
  vocab_size = vocab_size,
  batch_size = 1000,
  seq_len = 3,
  n_repeat = 4,
)


textenc = TransformerTextEncoder(split, vocab; startsym, endsym, unksym, padsym=unksym)
phen_creator = Creator(TransformerPhenotype, (;textenc=textenc))

weight_cache = WeightCache(maxsize=10_000_000_000)
counters = default_counters()
gene_counter = find(:type, AbstractGene, counters)
geno_creator = Creator(Delta, (Creator(Network, (rng, gene_counter, [
                          (Jevo.Transformer, trf_args),
                         ])),))
# Composite pop
pop_creator = Creator(Population, ("p", n_inds, PassThrough(geno_creator), PassThrough(phen_creator), counters))
env_creator = Creator(RepeatSequence, env_args)

abstract type OutputValue <: AbstractMetric end
abstract type MatchCount <: AbstractMetric end

function evaluate(individual)
  model = develop(phen_creator, individual)
  seq_str = infer(model, "1 2 3 1 2 3", max_len=15)[1]
  @info seq_str
  Jevo.play(env_creator(), [model])[1]
end

BestReporter() = create_op("Reporter",
        retriever=Jevo.get_individuals,
        operator=(s,is)->
          (m= StatisticalMeasurement(OutputValue, evaluate.(is), generation(s));
          @info m; @h5 m;))
BestVisualizer() = create_op("Reporter",
        retriever=Jevo.get_individuals,
        operator=(s,is)-> (@assert(length(is) == 1);
                           @info(string(is[1].id)* " "*visualize(is[1])))
)


state = if isfile(checkpointname)
          restore_from_checkpoint(checkpointname)
    else
        rm("statistics.h5", force=true)
        State("phylotest", rng, [pop_creator, env_creator], 
            [Checkpointer(checkpointname, interval=25),
            CreateMissingWorkers(4),
            InitializeAllPopulations(), 
            InitializePhylogeny(),
            InitializeDeltaCache(),
            UpdateParentsAcrossAllWorkers(time=true),
            SoloMatchMaker(["p"]), 
            Performer(time=true),
            ScalarFitnessEvaluator(["p"]), 
            TruncationSelector(k),
            BestReporter(),
            BestVisualizer(),
            CloneUniformReproducer(n_inds, time=true),
            Mutator(;mr=(0.1f0, 0.01f0, 0.001f0, 0.0001f0), n=n_weights_to_mutate, lookback=lookback, time=true),
            PopSizeAssertor(n_inds),
            UpdatePhylogeny(time=true),
            UpdateDeltaCache(time=true),
            ClearInteractionsAndRecords(),
           ], counters=counters)
      end

global_logger(JevoLogger())
run!(state, n_generations)
