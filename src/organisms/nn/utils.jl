export visualize
function mr_symbol(mr::Float32)
    mr == 1.0f0 && return "#"
    mr >= 0.1 && return "0"
    mr >= 0.01 && return "O"
    mr >= 0.001 && return "o"
    mr >= 0.0001 && return "."
end

get_weight_symbols(weights::Weights) = 
    lpad(string(weights.dims), 15) * " " *
        join([mr_symbol(w.mr) for w in weights.muts]) * "\n"
get_weight_symbols(factorized_weights::FactorWeight) =
    get_weight_symbols(factorized_weights.A) * get_weight_symbols(factorized_weights.B)
get_weight_symbols(pnr::PostNormResidual) = get_weight_symbols(pnr.layer)
get_weight_symbols(sa::SelfAttention) =
    "qkv\n" * get_weight_symbols(sa.qkv) *
    "out\n" * get_weight_symbols(sa.out)
get_weight_symbols(d::Dense) =
    get_weight_symbols(d.weights) * get_weight_symbols(d.bias)
get_weight_symbols(e::Embed) = get_weight_symbols(e.weights)
get_weight_symbols(e::EmbedDecoder) =
    get_weight_symbols(e.embed) * get_weight_symbols(e.bias)
get_weight_symbols(c::Chain) = 
    "chain\n" * join([get_weight_symbols(l) for l in c.layers])
get_weight_symbols(tdb::TransformerDecoderBlock) =
    get_weight_symbols(tdb.attention) * get_weight_symbols(tdb.ff)

get_weight_symbols(t::Transformer) = "Transformer\n" *
    "embed\n"* get_weight_symbols(t.embed) *
    "blocks\n" * join([get_weight_symbols(b) for b in t.blocks]) *
    "embeddecoder\n" * get_weight_symbols(t.embeddecoder) 

get_weight_symbols(network::Network) = join([get_weight_symbols(l) for l in network.layers])

visualize = get_weight_symbols
