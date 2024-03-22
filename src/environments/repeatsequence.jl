using Transformers.Datasets: batched
using Flux.Losses
export RepeatSequence, preprocess, infer
Base.@kwdef struct RepeatSequence <: AbstractEnvironment
    vocab_size::Int
    batch_size::Int
    seq_len::Int
    n_repeat::Int
end

function sample_sequence(vocab_size, seq_len, n_repeat)
    seq = rand(1:vocab_size, seq_len)
    repeat_seq = join(repeat(seq, n_repeat), " ")
    repeat_seq
end
function sample_batch(env::RepeatSequence)
    # Each string is enclosed in a tuple for the batch
    # If we were using encoder-decoder, we would have a tuple of two strings
    seqs = [(sample_sequence(env.vocab_size, env.seq_len, env.n_repeat),) for _ in 1:env.batch_size]
    batch = batched(seqs)
    batch[1] # get decoder batch

end

function shift_decode_loss(logits, trg, trg_mask)
    label = trg[:, 2:end, :]
    logitcrossentropy(@view(logits[:, 1:end-1, :]), label, trg_mask - 1)
end
function loss(input, trf)
    logits = trf(input)
    ce_loss = shift_decode_loss(logits, input.token, input.attention_mask)
    ce_loss
end
preprocess(trf::TransformerPhenotype, batch) = todevice(encode(trf.textenc, batch))

function infer(trf::TransformerPhenotype, str::String; max_len::Int=10, n_logits::Int=3, print_output::Bool=false)
    decoder_onehot = encode(trf.textenc, str).token
    decoder_tokens = decode(trf.textenc, decoder_onehot)
    seq = decoder_tokens[1:end-1]
    logits = []

    start_len = length(seq)
    for i in 1:max_len-start_len
        decoder_input = (token = todevice(lookup(trf.textenc, seq)),)
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


function step!(env::RepeatSequence, models::Vector{TransformerPhenotype})

    @assert length(models) == 1 "Only one model is supported for now"
    batch = sample_batch(env)
    trf = models[1]
    input_batch = preprocess(trf, batch)
    l = loss(input_batch, trf)
    [-l] # this framework maximizes fitness, so we report loss as negative fitness
end
(creator::Creator{RepeatSequence})(;kwargs...) = RepeatSequence(creator.kwargs...)
