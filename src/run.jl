function initialize!(state)
end
function run!(state::State, max_generations::Int)
    initialize!(state)
    for i in 1:max_generations
        operate!(state)
    end
end
