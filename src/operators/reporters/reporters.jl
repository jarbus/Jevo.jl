export Reporter
struct Reporter <: AbstractReporter
    condition::Function
    retriever::Function
    operator::Function
    updater::Function
    h5::Bool
    txt::Bool
    console::Bool
end
function Reporter(type::Type{<:AbstractMetric};
        condition=always,
        h5=true,
        txt=true,
        console=false)
    Reporter(condition,
             noop,
             (s,_)->measure(type, s, h5, txt, console),
             noop,
             h5, txt, console)
end
