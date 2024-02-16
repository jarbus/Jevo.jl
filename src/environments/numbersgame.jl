export CompareOnOne, AbstractNumbersGame
abstract type AbstractNumbersGame end
struct CompareOnOne <: AbstractNumbersGame end

function step!(environment::CompareOnOne, a::VectorPhenotype, b::VectorPhenotype)

    max_a_dim = argmax(a) 
    max_b_dim = argmax(b)
    a_score = a[max_b_dim] >= b[max_b_dim] ? 1.0 : 0.0
    b_score = b[max_a_dim] >= a[max_a_dim] ? 1.0 : 0.0
    return [a_score, b_score]
end

struct CompareOnAll <: AbstractNumbersGame end
