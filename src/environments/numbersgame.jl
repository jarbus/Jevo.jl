export CompareOnOne, AbstractNumbersGame
abstract type AbstractNumbersGame <: AbstractEnvironment end
struct CompareOnOne <: AbstractNumbersGame end

function step!(::CompareOnOne, phenotypes::Vector{VectorPhenotype})
    @assert length(phenotypes) == 2
    a, b = phenotypes
    max_a_dim = argmax(a.numbers) 
    max_b_dim = argmax(b.numbers)
    a_score = a.numbers[max_b_dim] >= b.numbers[max_b_dim] ? 1.0 : 0.0
    b_score = b.numbers[max_a_dim] >= a.numbers[max_a_dim] ? 1.0 : 0.0
    return [a_score, b_score]
end
(creator::Creator{CompareOnOne})(;kwargs...) = CompareOnOne()

struct CompareOnAll <: AbstractNumbersGame end
