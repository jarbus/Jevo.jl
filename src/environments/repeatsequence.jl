using Transformers.Datasets: batched
using Flux.Losses
export RepeatSequence, preprocess, infer, NegativeLoss
Base.@kwdef struct RepeatSequence <: AbstractEnvironment
    n_labels::Int
    batch_size::Int
    seq_len::Int
    n_repeat::Int
end
function split_into_strings(n)
    # Convert the input to a string and ensure it's exactly 3 characters long
    n_str = lpad(n, 3, '0')
    if length(n_str) != 3 || any(c -> !isdigit(c), n_str)
        throw(ArgumentError("Input must be a 3-digit integer or a valid string representation $n, $n_str"))
    end
    return n_str[1], n_str[2], n_str[3]
end

# ==== PERFORMANCE CRITICAL BEGIN (on server cpus, which are slow af
function sample_sequence(n_labels, seq_len, n_repeat, i)
    i_base = digits(i-1, base=n_labels) .|> string |> join
    seq = split_into_strings(i_base)
    concat_seq = join(seq, " ")
    repeat_seq = join((concat_seq for i = 1:n_repeat), " ")
    repeat_seq
end
# ==== PERFORMANCE CRITICAL END
function sample_batch(env::RepeatSequence)
    # Each string is enclosed in a tuple for the batch
    # If we were using encoder-decoder, we would have a tuple of two strings
    seqs = [(sample_sequence(env.n_labels, env.seq_len, env.n_repeat, i),) for i in 1:env.batch_size]
    batch = batched(seqs)
    batch[1] # get decoder batch
end

function shift_decode_loss(logits, trg, trg_mask::M) where M <: Transformers.NeuralAttentionlib.LengthMask
    n_tests = size(trg, 3)
    results = zeros(n_tests)
    for i in 1:n_tests  # (vocab_size, seq_len , batch_size)
        label = trg[:, 2:end, i:i]
        logits_view = @view(logits[:, 1:end-1, i:i])
        results[i] = -logitcrossentropy(logits_view, label, M(trg_mask.len[i:i]))
    end
    results
end

"""
    scores(input, scores, split_size, trf)

Computes losses in batches of size `split_size`. If a batch has a sufficiently lower loss than a good scorer, terminate early
"""
function scores(rng, input, scores, split_size, trf)
    ce_loss = fill(-Inf, size(scores))
    for idx in shuffle!(rng, collect(1:split_size:length(scores)))
        end_idx = idx+split_size-1
        split_input = (token = input.token[:,:,idx:end_idx], attention_mask = Transformers.NeuralAttentionlib.LengthMask(input.attention_mask.len[idx:end_idx]))
        split_logits = trf(split_input)
        #= split_input_token_sum = sum(input.token[:,:,idx:end_idx] |> Transformers.tocpudevice) =#
        #= tfr_param_sum = sum([sum(Transformers.tocpudevice(pv)) for pv in collect(Flux.params(trf.trf))]) =#
        #= @info "splitidx $idx: logits: $(sum(split_logits)) inputs: $(split_input_token_sum) params: $tfr_param_sum" =#
        split_ce_loss = shift_decode_loss(split_logits, split_input.token, split_input.attention_mask)
        ce_loss[idx:end_idx] .= split_ce_loss
        if sum(split_ce_loss) < sum(scores[idx:end_idx])-10std(scores[idx:end_idx])
            #= @info("skipping at idx $idx: $(sum(split_ce_loss)) < $(sum(scores[idx:end_idx])) - $(std(scores[idx:end_idx]))") =#
            return ce_loss
        end
    end
    if sum(ce_loss) .> sum(scores)
        scores[:] .= ce_loss[:]
    end
    ce_loss
end

function infer(trf::TransformerPhenotype, str::String; max_len::Int=10, n_logits::Int=3, print_output::Bool=false)
    decoder_onehot = encode(trf.textenc, str).token
    decoder_tokens = decode(trf.textenc, decoder_onehot)
    seq = decoder_tokens[1:end-1]
    logits = []

    start_len = length(seq)
    for i in 1:max_len-start_len
        decoder_input = (token = gpu(lookup(trf.textenc, seq)),)
        logit = trf(decoder_input)
        ntok = decode(trf.textenc, argmax(logit[:, end]))
        push!(seq, ntok)
        i <= n_logits && push!(logits, round.(softmax(Array(logit)[:,end]), digits=2))
        ntok == trf.textenc.endsym && break
    end
    seq_str = join(seq, " ")
    !print_output && return (seq_str, logits)
    println(seq_str)
    # print vocab
    for l in trf.textenc.vocab.list
        print(l, "\t")
    end
    println()
    # print logits
    for logit in logits
        for v in logit
            print(v, "\t")
        end
        println()
    end

end


function get_preprocessed_batch(env::RepeatSequence, tfr)
    if !isdefined(Main, :preprocessed_batch)
        @warn "Creating variable Main.preprocessed_batch"
        Main.preprocessed_batch = preprocess(tfr, sample_batch(env))
    end
    # There appears to be some memory management issue, where GPU OOMs.
    # Allocating a large amount of memory on the CPU appears to alleviate this 
    # issue. Garbage collection does not help. Unable to justify spending
    # more time on this, if it's resolved. On my laptop, this takes ~179μs per call
    size(zeros(1_000_000)) # TODO see if we can change this to fill(undef, 1_000_000)
    Main.preprocessed_batch |> deepcopy |> gpu
end

function get_best_scores(env::RepeatSequence)
    if !isdefined(Main, :best_scores)
        @warn "Creating variable Main.best_scores"
        Main.best_scores = fill(-Inf, env.batch_size)
    end
    Main.best_scores
end

function play(env::RepeatSequence, ids::Vector{Int}, models::Vector{TransformerPhenotype})
    @assert length(models) == 1 "Only one model is supported for now"
    @assert env.seq_len > 1
    #= @info("evaluating $(ids[1])") =#
    tfr = models[1]
    input_batch = get_preprocessed_batch(env, tfr)
    best_scores = get_best_scores(env)
    split_size = env.n_labels ^ (env.seq_len) # basically batch size
    results = scores(StableRNG(ids[1]), input_batch, best_scores, split_size, tfr)
    [Interaction(ids[1], [test_idx], r) for (test_idx, r) in enumerate(results)]
end

abstract type NegativeLoss <: AbstractMetric end

function evaluate(env_creator::Creator{RepeatSequence}, individual::Individual)
  percent_correct = fetch(@spawnat(2, begin
    function percentage_evaluation_npeat(trf::TransformerPhenotype; n::Int, kwargs...)
        n_perfect = 0
        for i in 0:n-1, j in 0:n-1, k in 0:n-1
            prompt = "$i $j $k $i $j $k"
            full_str = infer(trf, prompt; kwargs...)[1]
            if length(full_str) >= 27
              n_perfect += full_str[5:15] == full_str[17:27]
              @info full_str
            else
            end
        end
        n_perfect / n^3
    end
    model = develop(individual.developer, individual)
    percentage_evaluation_npeat(model, n=env_creator.kwargs.n_labels, max_len=15)
  end))
  @info "Percentage perfect: $(round(percent_correct, digits=3))"
  fetch(@spawnat(2, begin
      device!(Main.jevo_device_id)
      mean(interaction.score for interaction in Jevo.play(env_creator, [individual]))
    end))
end
