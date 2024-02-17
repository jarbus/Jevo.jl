export Counter, inc!, value
export default_counters

mutable struct Counter <: AbstractCounter
    type::Type
    current_value::Int
    lock::ReentrantLock
end

Base.show(io::IO, c::Counter) = print(io, "Counter($(c.type), $(c.current_value))")

Counter(type::Type) = Counter(type, 1, Threads.ReentrantLock())
value(c::Counter) = c.current_value

default_counters() = [Counter(AbstractGene),
                      Counter(AbstractIndividual),
                      Counter(AbstractGeneration),
                      Counter(AbstractMatch),
                     ]

function inc!(counter::Counter)
    lock(counter.lock) do
        value = counter.current_value
        counter.current_value += 1
        return value
    end
end

function inc!(c::Counter, n::Int)
    lock(c.lock) do
        values = [inc!(c) for _ in 1:n]
        return values
    end
end

