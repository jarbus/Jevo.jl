export NSGAIISelector


@define_op "NSGAIISelector" "AbstractEvaluator"
NSGAIISelector(n_parents::Int, ids::Vector{String}=String[]; kwargs...) =
    create_op("NSGAIISelector",
    retriever=PopulationRetriever(ids),
    updater=map(map((s,p)->choose_nsgaii_parents!(s, p, n_parents))),
    ;kwargs...)


function choose_nsgaii_parents!(s::AbstractState, pop::Population, n_parents::Int)
    outcomes = getonly(x->x isa OutcomeMatrix, pop.data).matrix
    selected_indices = nsga2(outcomes, n_parents, generation(s))
    selected_parents = pop.individuals[selected_indices]
    pop.individuals = selected_parents
end

# In maximization, a dominates b if every objective in a is >= that in b, and at least one is >.
function dominates(a::AbstractVector, b::AbstractVector)
    return all(a .>= b) && any(a .> b)
end

function nsga2(outcomes::Matrix{Float64}, p::Int, gen=nothing)
    N, n_obj = size(outcomes)
    
    # Initialize domination sets and counts.
    S = [Int[] for _ in 1:N]
    n = zeros(Int, N)
    rank = zeros(Int, N)
    fronts = Vector{Vector{Int}}()
    push!(fronts, Int[])
    
    # Compute domination relationships.
    for i in 1:N
        equal = Vector{Int}()
        for j in 1:N
            if i == j
                continue
            end
            if dominates(outcomes[i, :], outcomes[j, :])
                push!(S[i], j)
            elseif dominates(outcomes[j, :], outcomes[i, :])
                n[i] += 1
            elseif outcomes[i, :] == outcomes[j, :]
                push!(equal, j)
            end
        end
        if n[i] == 0 && (isempty(equal) || i < minimum(equal))
            rank[i] = 1
            push!(fronts[1], i)
        end
    end
    @assert length(fronts[1]) > 0 "No individuals found in the first front"
    if !isnothing(gen) 
        m = Measurement("n_pareto_front", length(fronts[1]), gen)
        @info m
        @h5 m
    end
    
    # Generate subsequent fronts.
    currentFront = 1
    while !isempty(fronts[currentFront])
        Q = Int[]
        for i in fronts[currentFront]
            for j in S[i]
                n[j] -= 1
                if n[j] == 0
                    rank[j] = currentFront + 1
                    push!(Q, j)
                end
            end
        end
        if !isempty(Q)
            push!(fronts, Q)
        else
            break
        end
        currentFront += 1
    end
    
    # Crowding distance assignment.
    distances = zeros(Float64, N)
    for front in fronts
        l = length(front)
        if l == 0
            continue
        end
        for i in front
            distances[i] = 0.0
        end
        for m in 1:n_obj
            # For maximization, sort descending.
            sorted_front = sort(front, by = i -> outcomes[i, m], rev = true)
            distances[sorted_front[1]] = Inf
            distances[sorted_front[end]] = Inf
            obj_max = outcomes[sorted_front[1], m]
            obj_min = outcomes[sorted_front[end], m]
            range = obj_max - obj_min
            if range == 0
                continue
            end
            for k in 2:(l-1)
                distances[sorted_front[k]] += (outcomes[sorted_front[k-1], m] - outcomes[sorted_front[k+1], m]) / range
            end
        end
    end

    # Sort individuals by increasing rank and, within same rank, by descending crowding distance.
    sorted_indices = sort(1:N, by = i -> (rank[i], -distances[i]))
    
    return sorted_indices[1:p]
end
