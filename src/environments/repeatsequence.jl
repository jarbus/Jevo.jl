using Transformers.Datasets: batched
using Flux.Losses
export RepeatSequence, preprocess, infer
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

function shift_decode_loss(logits, trg, trg_mask)
    n_tests = size(trg, 1)
    results = zeros(n_tests)
    for i in 1:n_tests
        label = trg[i:i, 2:end, :]
        logits_view = @view(logits[i:i, 1:end-1, :])
        results[i] = -logitcrossentropy(logits_view, label, trg_mask - 1)
    end
    results
end

function scores(input, trf)
    logits = trf(input)
    ce_loss = shift_decode_loss(logits, input.token, input.attention_mask)
    ce_loss
end

preprocess(trf::TransformerPhenotype, batch) = encode(trf.textenc, batch)

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


function get_preprocessed_batch(env, tfr)
    # get global variable Main.weight_cache for weight cache
    # check if weight_cache is defined
    if !isdefined(Main, :preprocessed_batch)
        @warn "Creating variable Main.preprocessed_batch"
        Main.preprocessed_batch = preprocess(tfr, sample_batch(env))
    end
    # There appears to be some memory management issue, where GPU OOMs.
    # Allocating a large amount of memory on the CPU appears to alleviate this 
    # issue. Garbage collection does not help. Unable to justify spending
    # more time on this, if it's resolved.
    size(zeros(1_000_000))
    Main.preprocessed_batch |> deepcopy |> gpu
end

(creator::Creator{RepeatSequence})(;kwargs...) = RepeatSequence(creator.kwargs...)

function play(env::RepeatSequence, ids::Vector{Int}, models::Vector{TransformerPhenotype})
    @assert length(models) == 1 "Only one model is supported for now"
    tfr = models[1]
    input_batch = get_preprocessed_batch(env, tfr)
    results = scores(input_batch, tfr)
    [Interaction(ids[1], [test_idx], r) for (test_idx, r) in enumerate(results)]
end
