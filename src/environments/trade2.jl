using Images

export TradeGridWorld, render

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
    view_radius::Int # Radius of player's view window
end

function TradeGridWorld(n::Int, p::Int; max_steps::Int=100, view_radius::Int=30)
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
    TradeGridWorld(n, p, grid_apples, grid_bananas, players, 1, max_steps, view_radius)
end

function step!(env::TradeGridWorld, ids::Vector{Int}, phenotypes::Vector{P}) where P<:AbstractPhenotype
    @assert length(ids) == length(phenotypes) == env.p
    interactions = Interaction[]
    observations = make_observations(env, ids, phenotypes)
    actions = get_actions(observations, phenotypes)
    for (i, id) in enumerate(ids)
        player = env.players[i]
        action_values = actions[i]
        @assert length(action_values) == 4
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
        elseif place_action > 0  # Place bananas
            amount = min(player.resource_bananas, abs(place_action))
            player.resource_bananas -= amount
            env.grid_bananas[grid_x, grid_y] += amount
        end
        if pick_action < 0 && env.grid_apples[grid_x, grid_y] > 0
            amount = min(env.grid_apples[grid_x, grid_y], abs(pick_action))
            player.resource_apples += amount
            env.grid_apples[grid_x, grid_y] -= amount
        elseif pick_action > 0 && env.grid_bananas[grid_x, grid_y] > 0
            amount = min(env.grid_bananas[grid_x, grid_y], pick_action)
            player.resource_bananas += amount
            env.grid_bananas[grid_x, grid_y] -= amount
        end

        score = log(player.resource_apples) + log(player.resource_bananas)
        push!(interactions, Interaction(id, [], score))

        env.players[i] = player  # Update player state
    end
    env.step_counter += 1
    return interactions
end

done(env::TradeGridWorld) = env.step_counter > env.max_steps

# Creates RGB pixel observations for each player in the environment.
function make_observations(env::TradeGridWorld, ids::Vector{Int}, phenotypes::Vector{P}) where P<:AbstractPhenotype
    base_img = render(env)
    view_radius = env.view_radius
    view_size = 2 * view_radius + 1
    observations = Vector{Array{RGB{N0f8}, 2}}(undef, length(env.players))
    
    for (i, player) in enumerate(env.players)
        obs = fill(RGB{N0f8}(1, 1, 1), view_size, view_size)
        px = round(Int, player.position[1]) + 1
        py = round(Int, player.position[2]) + 1
        
        # Calculate ranges for source and destination
        x_src_range = max(1, px - view_radius):min(env.n, px + view_radius)
        y_src_range = max(1, py - view_radius):min(env.n, py + view_radius)
        x_dst_start = view_radius + 1 - (px - first(x_src_range))
        y_dst_start = view_radius + 1 - (py - first(y_src_range))
        x_dst_range = x_dst_start:(x_dst_start + length(x_src_range) - 1)
        y_dst_range = y_dst_start:(y_dst_start + length(y_src_range) - 1)
        
        # Single array operation instead of nested loops
        obs[x_dst_range, y_dst_range] = view(base_img, x_src_range, y_src_range)
        observations[i] = obs
    end
    
    observations
end

function get_actions(observations::Vector, phenotypes::Vector{P}) where P<:AbstractPhenotype
    [pheno(obs) for (obs, pheno) in zip(observations, phenotypes)]
end

player_colors = [RGB(0.67, 0.87, 0.73), RGB(0.47, 0.60, 0.54)]

function render(env::TradeGridWorld)
    n = env.n
    img = Array{RGB{N0f8}}(undef, n, n)
    fill!(img, RGB{N0f8}(0, 0, 0))

    for idx in eachindex(env.players)
        player = env.players[idx]
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
                    img[x, y] = player_colors[idx]
                end
            end
        end
    end
    # Render apples and bananas on the grid
    for x in 1:n, y in 1:n
        if env.grid_apples[x, y] > 0
            img[x, y] = RGB{N0f8}(env.grid_apples[x,y]/10, 0, 0)  # Red for apples
        elseif env.grid_bananas[x, y] > 0
            img[x, y] = RGB{N0f8}(0, env.grid_bananas[x,y]/10, 0)  # Green for bananas
        end
    end
    img
end
