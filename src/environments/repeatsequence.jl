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

# logit idx where evaluation inference begins
get_decode_start_idx(env::RepeatSequence) = (2*env.seq_len) + 1

function shift_decode_loss(logits, trg, trg_mask::M, decode_start_idx) where M <: Transformers.NeuralAttentionlib.LengthMask
    # ignore start
    label = trg[:, (decode_start_idx+1):end, :]
    # ignore end sequence, idx doesn't correspond to <s>
    -logitcrossentropy(@view(logits[:, decode_start_idx:end-1, :]), label, trg_mask-1)
end

function loss(input, textmodel, decode_start_idx)
    logits = textmodel(input)
    clamp!(logits, -10_000f0, 10_000f0)
    ce_loss = shift_decode_loss(logits, input.token, input.attention_mask, decode_start_idx)
    ce_loss
end


function infer(tm::TextModel, str::String; max_len::Int=10, n_logits::Int=3, print_output::Bool=false)
    decoder_onehot = encode(tm.textenc, str).token
    decoder_tokens = decode(tm.textenc, decoder_onehot)
    seq = decoder_tokens[1:end-1]
    logits = []

    start_len = length(seq)
    for i in 1:max_len-start_len
        decoder_input = (token = gpu(lookup(tm.textenc, seq)),)
        logit = tm(decoder_input)
        ntok = decode(tm.textenc, argmax(logit[:, end]))
        push!(seq, ntok)
        i <= n_logits && push!(logits, round.(softmax(Array(logit)[:,end]), digits=2))
        ntok == tm.textenc.endsym && break
    end
    seq_str = join(seq, " ")
    !print_output && return (seq_str, logits)
    println(seq_str)
    # print vocab
    for l in tm.textenc.vocab.list
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


function get_preprocessed_batch(env::RepeatSequence, tm::TextModel)
    if !isdefined(Main, :preprocessed_batch)
        @warn "Creating variable Main.preprocessed_batch"
        Main.preprocessed_batch = encode(tm.textenc, sample_batch(env))
    end
    # There appears to be some memory management issue, where GPU OOMs.
    # Allocating a large amount of memory on the CPU appears to alleviate this 
    # issue. Garbage collection does not help. Unable to justify spending
    # more time on this, if it's resolved. On my laptop, this takes ~179μs per call
    size(zeros(100_000))
    Main.preprocessed_batch |> deepcopy |> gpu
end

function step!(env::RepeatSequence, ids::Vector{Int}, models::Vector{TextModel{TE,M}}) where {TE, M}
    @assert length(models) == 1 "Only one model is supported for now"
    @assert env.seq_len > 1
    model = models[1]
    input_batch = get_preprocessed_batch(env, model)
    decode_start_idx = get_decode_start_idx(env)
    ce_loss = loss(input_batch, model, decode_start_idx)
    [Interaction(ids[1], [], ce_loss)]
end

abstract type NegativeLoss <: AbstractMetric end

function evaluate(env_creator::Creator{RepeatSequence}, individual::Individual)
  wid = workers()[1]
  percent_correct = fetch(@spawnat(wid, begin
    function percentage_evaluation_npeat(tm::TextModel; n::Int, kwargs...)
        n_perfect = 0
        for i in 0:n-1, j in 0:n-1, k in 0:n-1
            prompt = "$i $j $k $i $j $k"
            full_str = infer(tm, prompt; kwargs...)[1]
            if length(full_str) >= 27
              n_perfect += full_str[5:15] == full_str[17:27]
              @info full_str
            else
            end
        end
        n_perfect / n^3
    end
    model = develop(individual.developer, individual)
  percent_correct =  percentage_evaluation_npeat(model, n=env_creator.kwargs.n_labels, max_len=15)
  end))
  @info "Percentage perfect: $(round(percent_correct, digits=3))"
  fetch(@spawnat(wid, begin
      device!(Main.jevo_device_id)
      mean(interaction.score for interaction in Jevo.play(env_creator, [individual]))
    end))
end
