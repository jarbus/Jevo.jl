using Images
using FileIO

export TradeGridWorld, run_random_episode

struct DummyPhenotype <: AbstractPhenotype
    numbers::Vector{Float64}
end

mutable struct PlayerState
    id::Int
    position::Tuple{Float64, Float64}
    resource_apples::Float64
    resource_bananas::Float64
end

abstract type AbstractGridworld <: Jevo.AbstractEnvironment end

mutable struct TradeGridWorld <: AbstractGridworld
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
        if i % 2 == 1
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

function step!(env::TradeGridWorld, ids::Vector{Int}, phenotypes::Vector{DummyPhenotype})
    @assert length(ids) == length(phenotypes) == env.p
    interactions = Interaction[]
    for (i, id) in enumerate(ids)
        player = env.players[i]
        phenotype = phenotypes[i]
        action_values = get_action_values(phenotype)
        @assert length(action_values) == 3
        dx, dy, place_action, pick_action = action_values
        # Update player position
        new_x = clamp(player.position[1] + dx, 0, env.n - 1)
        new_y = clamp(player.position[2] + dy, 0, env.n - 1)
        player.position = (new_x, new_y)
        # Resource action
        grid_x = clamp(floor(Int, new_x) + 1, 1, env.n)
        grid_y = clamp(floor(Int, new_y) + 1, 1, env.n)
        if place_action < 0  # Place apples
            amount = min(player.resource_apples, abs(place_action))
            player.resource_apples -= amount
            env.grid_apples[grid_x, grid_y] += amount
        else  # Place bananas
            amount = min(player.resource_bananas, abs(place_action))
            player.resource_bananas -= amount
            env.grid_bananas[grid_x, grid_y] += amount
        end
        # Players can pick up resources from the grid (logic can be added here)
        if pick_action > 0 && env.grid_apples[grid_x, grid_y] > 0
            player.resource_apples += env.grid_apples[grid_x, grid_y]
            env.grid_apples[grid_x, grid_y] = 0
        elseif pick_action < 0 && env.grid_bananas[grid_x, grid_y] > 0
            player.resource_bananas += env.grid_bananas[grid_x, grid_y]
            env.grid_bananas[grid_x, grid_y] = 0
        end

        score = log(player.resource_apples) + log(player.resource_bananas)
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

function run_random_episode(;n::Int=10, p::Int=2, max_steps::Int=100, output_filename::String="episode.gif")
    env = TradeGridWorld(n, p, max_steps=max_steps)
    frames = []
    while !done(env)
        ids = [player.id for player in env.players]
        # Generate random actions for each player
        phenotypes = [DummyPhenotype(randn(4)) for _ in env.players]
        step!(env, ids, phenotypes)
        push!(frames, render(env))
    end
    frames = cat(frames..., dims=3)
    println(size(frames))
    save(output_filename, frames)
end

function render(env::TradeGridWorld)
    n = env.n
    img = Array{RGB{N0f8}}(undef, n, n)
    fill!(img, RGB{N0f8}(0, 0, 0))
    # Render players as blue circles of radius 4
    for player in env.players
        x_center = player.position[1] + 1  # Adjust for 1-based indexing
        y_center = player.position[2] + 1  # Adjust for 1-based indexing
        radius = 4  # Circle radius

        # Determine the bounding box for the circle
        x_min = max(floor(Int, x_center - radius), 1)
        x_max = min(ceil(Int, x_center + radius), n)
        y_min = max(floor(Int, y_center - radius), 1)
        y_max = min(ceil(Int, y_center + radius), n)

        for x in x_min:x_max
            for y in y_min:y_max
                # Compute the distance from the center
                dx = x - x_center
                dy = y - y_center
                distance = sqrt(dx^2 + dy^2)
                if distance <= radius
                    img[x, y] = RGB{N0f8}(0, 0, 1)  # Blue color for agents
                end
            end
        end
    end
    # Render apples and bananas on the grid
    for x in 1:n, y in 1:n
        if env.grid_apples[x, y] > 0
            img[x, y] = RGB{N0f8}(1, 0, 0)  # Red for apples
        elseif env.grid_bananas[x, y] > 0
            img[x, y] = RGB{N0f8}(0, 1, 0)  # Green for bananas
        end
    end
    img
end
