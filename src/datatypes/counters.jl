export Counter, inc!, value
export default_counters

"""
    Counter(type::Type)

Holds an integer to increase by 1 with `inc!(counter::Counter)`. Used for tracking gene ids, generations, individual ids, etc. Counters have a default value of `1`.
"""
mutable struct Counter <: AbstractCounter
    type::Type
    current_value::Int
    lock::ReentrantLock
end
Counter(type::Type) = Counter(type, 1, Threads.ReentrantLock())

Base.show(io::IO, c::Counter) = print(io, "Counter($(c.type), $(c.current_value))")

value(c::Counter) = c.current_value

default_counters() = [Counter(AbstractGene),
                      Counter(AbstractIndividual),
                      Counter(AbstractGeneration),
                      Counter(AbstractMatch),
                     ]

"""
    inc!(counter::Counter, n::Int=1)

Increment the value of the given `counter` by `n` in a thread-safe manner.
Returns the value of before the increment.
"""
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

