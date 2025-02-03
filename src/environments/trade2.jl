using Images
using FileIO

export TradeGridWorld, run_random_episode

struct DummyPhenotype <: AbstractPhenotype
    numbers::Vector{Float64}
end

struct PlayerState
    id::Int
    position::Tuple{Float64, Float64}
    resource_apples::Float64
    resource_bananas::Float64
end

abstract type AbstractGridworld <: Jevo.AbstractEnvironment end
struct TradeGridWorld <: AbstractGridworld
    n::Int           # Grid size
    p::Int           # Number of players
    grid_apples::Array{Float64,2}
    grid_bananas::Array{Float64,2}
    players::Vector{PlayerState}
    step_counter::Int
    max_steps::Int
end

function TradeGridWorld(n::Int, p::Int; max_steps::Int=100)
    grid_apples = zeros(n, n)
    grid_bananas = zeros(n, n)
    players = PlayerState[]
    for i in 1:p
        # Players start with 10 of one resource and zero of the other
        if rand() < 0.5
            apples = 10.0
            bananas = 0.0
        else
            apples = 0.0
            bananas = 10.0
        end
        position = (rand() * n, rand() * n)
        push!(players, PlayerState(i, position, apples, bananas))
    end
    TradeGridWorld(n, p, grid_apples, grid_bananas, players, 0, max_steps)
end

function step!(env::TradeGridWorld, ids::Vector{Int}, phenotypes::Vector{AbstractPhenotype})
    @assert length(ids) == length(phenotypes) == env.p
    interactions = Interaction[]
    for (i, id) in enumerate(ids)
        player = env.players[i]
        phenotype = phenotypes[i]
        action_values = get_action_values(phenotype)
        @assert length(action_values) == 3
        dx, dy, resource_action = action_values
        # Update player position
        new_x = mod(player.position[1] + dx, env.n)
        new_y = mod(player.position[2] + dy, env.n)
        player.position = (new_x, new_y)
        # Resource action
        grid_x = clamp(floor(Int, new_x) + 1, 1, env.n)
        grid_y = clamp(floor(Int, new_y) + 1, 1, env.n)
        if resource_action < 0  # Place apples
            amount = -resource_action
            if player.resource_apples >= amount
                player.resource_apples -= amount
                env.grid_apples[grid_x, grid_y] += amount
            end
        else  # Place bananas
            amount = resource_action
            if player.resource_bananas >= amount
                player.resource_bananas -= amount
                env.grid_bananas[grid_x, grid_y] += amount
            end
        end
        # Players can pick up resources from the grid (logic can be added here)

        # Record interaction
        score = 0.0  # Define scoring mechanism if needed
        push!(interactions, Interaction(id, [], score))

        env.players[i] = player  # Update player state
    end
    env.step_counter += 1
    return interactions
end

done(env::TradeGridWorld) = env.step_counter >= env.max_steps

function get_action_values(phenotype::AbstractPhenotype)
    # Assuming the phenotype provides action values as numbers
    return phenotype.numbers
end

function run_random_episode()
    env = TradeGridWorld(10, 2, max_steps=100)
    frames = []
    while !done(env)
        ids = [player.id for player in env.players]
        # Generate random actions for each player
        phenotypes = [DummyPhenotype(randn(3)) for _ in env.players]
        step!(env, ids, phenotypes)
        push!(frames, render(env))
    end
    # Save the frames as a GIF
    save("episode.gif", cat(frames..., dims=4))
end

function render(env::TradeGridWorld)
    n = env.n
    img = fill(RGB{N0f8}(0, 0, 0), n, n)
    # Render apples and bananas on the grid
    for x in 1:n, y in 1:n
        if env.grid_apples[x, y] > 0
            img[x, y] = RGB{N0f8}(1, 0, 0)  # Red for apples
        elseif env.grid_bananas[x, y] > 0
            img[x, y] = RGB{N0f8}(0, 1, 0)  # Green for bananas
        end
    end
    # Render players as blue circles
    for player in env.players
        x = clamp(floor(Int, player.position[1]) + 1, 1, n)
        y = clamp(floor(Int, player.position[2]) + 1, 1, n)
        img[x, y] = RGB{N0f8}(0, 0, 1)
    end
    img
end
