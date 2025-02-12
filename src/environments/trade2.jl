using Images
using Images.ImageCore: colorview, RGB

export TradeGridWorld, render, LogTradeRatios, ClearTradeRatios 

const SELF_COLOR = [0.67f0, 0.87f0, 0.73f0]
const OTHER_COLOR = [0.47f0, 0.60f0, 0.54f0]
const PLAYER_RADIUS = 4
const STARTING_RESOURCES = 10f0
const POOL_REWARD = 0.05f0  # Reward for standing in water pool
const POOL_COLOR = [0.4f0, 0.8f0, 1.0f0]  # Blue color for water
const FOOD_BONUS_EPSILON = 0.1

struct TradeRatio <: AbstractMetric end
struct PrimaryResourceCount <: AbstractMetric end
struct SecondaryResourceCount <: AbstractMetric end

abstract type AbstractGridworld <: Jevo.AbstractEnvironment end

mutable struct PlayerState
    id::Int
    position::Tuple{Float64, Float64}
    resource_apples::Float64
    resource_bananas::Float64
end

struct TradeRatioInteraction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    trade_ratio::Float64
end

struct PrimaryResourceCountInteraction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    count::Float64
end

struct SecondaryResourceCountInteraction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    count::Float64
end

mutable struct TradeGridWorld <: AbstractGridworld
    n::Int           # Grid size
    p::Int           # Number of players
    grid_apples::Array{Float64,2}
    grid_bananas::Array{Float64,2}
    players::Vector{PlayerState}
    step_counter::Int
    max_steps::Int
    view_radius::Int # Radius of player's view window
    reset_interval::Int  # Reset map every N steps
    pool_radius::Int # Radius of central water pool
    render_filename::String
    frames::Vector{Array{Float32,3}}
    perspective_frames::Vector  # each player's obs 
end

function TradeGridWorld(n::Int, p::Int, max_steps::Int=100, view_radius::Int=30, reset_interval::Int=25, pool_radius::Int=5, render_filename::String="")
    @assert max_steps % reset_interval == 0
    grid_apples = zeros(n, n)
    grid_bananas = zeros(n, n)
    players = PlayerState[]
    # start player 1 in top left corner, player 2 in bottom right corner
    player_offsets = Float32[6, n-6]
    for i in 1:p
        # Players start with 10 of one resource and zero of the other
        position = (player_offsets[i], player_offsets[i])
        push!(players, PlayerState(i, position, 0.0, 0.0))
    end
    env = TradeGridWorld(n, p, grid_apples, grid_bananas, players, 1, max_steps, view_radius, reset_interval, pool_radius, render_filename, Array{Float32, 3}[], [Array{Float32, 3}[] for i in 1:p])
    reset_map!(env)
    env
end

function log_trade_ratio(state, individuals, h5)
    # extract all trade ratio interactions
    ratios = [int.trade_ratio for ind in individuals for int in ind.interactions if int isa TradeRatioInteraction]
    primaries = [int.count for ind in individuals for int in ind.interactions if int isa PrimaryResourceCountInteraction]
    secondaries = [int.count for ind in individuals for int in ind.interactions if int isa SecondaryResourceCountInteraction]
    ratio_m=StatisticalMeasurement(TradeRatio, ratios, generation(state))
    primary_m=StatisticalMeasurement(PrimaryResourceCount, primaries, generation(state))
    secondary_m=StatisticalMeasurement(SecondaryResourceCount, secondaries, generation(state))
    @info(ratio_m)
    @info(primary_m)
    @info(secondary_m)
    h5 && @h5(ratio_m)
    h5 && @h5(primary_m)
    h5 && @h5(secondary_m)
    individuals
end

function remove_trade_ratios!(state, individuals)
    for ind in individuals
        filter!(interaction->interaction isa Interaction, ind.interactions)
    end
end

LogTradeRatios(;h5=true, kwargs...) = create_op("Reporter";
    retriever=get_individuals,
    operator=(s,is)->log_trade_ratio(s,is, h5),
    updater=remove_trade_ratios!,kwargs...)

ClearTradeRatios(;kwargs...) = create_op("Reporter";
    retriever=get_individuals,
    updater=remove_trade_ratios!,kwargs...)

# 0 when ratio is very unbalanced, 0.5 when ratio is even
function inverse_absolute_difference(apples, bananas)
    max_resource = max(apples, bananas)
    min_resource = min(apples, bananas)
    ratio =  max_resource / min_resource
    isnan(ratio) && return 0
    return 1 / (ratio + 1)
end
inverse_absolute_difference(player::PlayerState) = inverse_absolute_difference(player.resource_apples, player.resource_bananas)

function collect_nearby_resources(grid::Array{Float64, 2}, player::PlayerState, amount)
    x, y = player.position
    x_min = max(1, floor(Int, x - PLAYER_RADIUS))
    x_max = min(size(grid, 1), ceil(Int, x + PLAYER_RADIUS))
    y_min = max(1, floor(Int, y - PLAYER_RADIUS))
    y_max = min(size(grid, 2), ceil(Int, y + PLAYER_RADIUS))

    total_collected = 0.0
    for i in x_min:x_max, j in y_min:y_max 
        if grid[i,j] > 0 && (i - x)^2 + (j - y)^2 <= PLAYER_RADIUS^2
            amount_to_collect = min(amount - total_collected, grid[i, j]);
            grid[i, j] -= amount_to_collect;
            total_collected += amount_to_collect;
            if total_collected >= amount 
                return total_collected 
            end
        end
    end
    return total_collected
end

function step!(env::TradeGridWorld, ids::Vector{Int}, phenotypes::Vector{P}) where P<:AbstractPhenotype

    @assert length(ids) == length(phenotypes) == env.p
    
    interactions = []

    if env.step_counter > 1 && env.step_counter % env.reset_interval == 1
        append!(interactions, make_resource_interactions(env, ids))
        reset_map!(env)
    end
    observations = make_observations(env, ids, phenotypes)

    if !isempty(env.render_filename) 
        push!(env.frames, render(env, 1))  # Always render from player 1's perspective
        for i in 1:env.p
            # observations have a batch size associated with them, so we need to remove that
            push!(env.perspective_frames[i], observations[i][:,:,:,1])
        end
    end

    actions = get_actions(observations, phenotypes)
    for (i, id) in enumerate(ids)
        player = env.players[i]
        action_values = actions[i]
        @assert length(action_values) == 4
        # assert all actions are not nan or inf
        @assert !any(isnan.(action_values)) && !any(isinf.(action_values))
        dx, dy, place_action, pick_action = action_values
        dx = clamp(dx, -2, 2)
        dy = clamp(dy, -2, 2)
        # Update player position
        new_x = clamp(player.position[1] + dx, 1, env.n)
        new_y = clamp(player.position[2] + dy, 1, env.n)
        prev_pos = player.position
        player.position = (new_x, new_y)
        # Resource action
        try
            grid_x = clamp(floor(Int, new_x) + 1, 1, env.n)
            grid_y = clamp(floor(Int, new_y) + 1, 1, env.n)
        catch e
            @error "new_x: $new_x, new_y: $new_y, dx: $dx, dy: $dy, prev_pos: $prev_pos, error: $e"
        end
        grid_x = clamp(floor(Int, new_x) + 1, 1, env.n)
        grid_y = clamp(floor(Int, new_y) + 1, 1, env.n)
        # Determine primary/secondary resources based on player number
        is_player_one = i % 2 == 1
        
        # Handle placing resources
        if place_action > 0  # Place primary resource
            if is_player_one
                amount = min(player.resource_apples, place_action)
                player.resource_apples -= amount
                env.grid_apples[grid_x, grid_y] += amount
            else
                amount = min(player.resource_bananas, place_action)
                player.resource_bananas -= amount
                env.grid_bananas[grid_x, grid_y] += amount
            end
        elseif place_action < 0  # Place secondary resource
            if is_player_one
                amount = min(player.resource_bananas, abs(place_action))
                player.resource_bananas -= amount
                env.grid_bananas[grid_x, grid_y] += amount
            else
                amount = min(player.resource_apples, abs(place_action))
                player.resource_apples -= amount
                env.grid_apples[grid_x, grid_y] += amount
            end
        end

        if pick_action > 0  # Pick primary resource
            if is_player_one
                player.resource_apples += collect_nearby_resources(env.grid_apples, player, pick_action)
            else
                player.resource_bananas += collect_nearby_resources(env.grid_bananas, player, pick_action)
            end
        elseif pick_action < 0  # Pick secondary resource
            if is_player_one
                player.resource_bananas += collect_nearby_resources(env.grid_bananas, player, abs(pick_action))
            else
                player.resource_apples += collect_nearby_resources(env.grid_apples, player, abs(pick_action))
            end
        end

        # Check if player is in water pool
        center = env.n ÷ 2
        dx = player.position[1] - center
        dy = player.position[2] - center
        in_pool = sqrt(dx^2 + dy^2) <= env.pool_radius
        pool_bonus = in_pool ? POOL_REWARD : 0.0f0
        
        score = (player.resource_apples + FOOD_BONUS_EPSILON) * (player.resource_bananas + FOOD_BONUS_EPSILON) + pool_bonus
        push!(interactions, Interaction(id, [], score))

        env.players[i] = player  # Update player state
    end
    env.step_counter += 1
    # Record trade ratio
    if done(env)
        if !isempty(env.render_filename)
            push!(env.frames, render(env, 1))  # Always render from player 1's perspective
            root = split(env.render_filename, ".")[1]
            save_gif(env.frames, "$(root)_full.gif")

            for i in 1:env.p
                save_gif(env.perspective_frames[i], "$(root)_player_$(i).gif")
            end

        end
    end
    return interactions
end

done(env::TradeGridWorld) = env.step_counter > env.max_steps

function make_resource_interactions(env::TradeGridWorld, ids::Vector{Int})
    interactions = []
    @assert env.p == 2
    for i in 1:env.p, j in (i+1):env.p
        push!(interactions, 
            TradeRatioInteraction(ids[i], [ids[j]], inverse_absolute_difference(env.players[i])),
            TradeRatioInteraction(ids[j], [ids[i]], inverse_absolute_difference(env.players[j])))
        # compute primary
        push!(interactions, 
            PrimaryResourceCountInteraction(ids[i], [ids[j]], env.players[i].resource_apples),
            PrimaryResourceCountInteraction(ids[j], [ids[i]], env.players[j].resource_bananas))
        # compute secondary
        push!(interactions, 
            SecondaryResourceCountInteraction(ids[i], [ids[j]], env.players[i].resource_bananas),
            SecondaryResourceCountInteraction(ids[j], [ids[i]], env.players[j].resource_apples))
    end
    interactions
end

# Creates RGB pixel observations for each player in the environment.
function make_observations(env::TradeGridWorld, ids::Vector{Int}, phenotypes::Vector{P}) where P<:AbstractPhenotype
    base_img = render(env)
    view_radius = env.view_radius
    view_size = 2 * view_radius + 1
    observations = Vector(undef, length(env.players))
    
    for (i, player) in enumerate(env.players)
        # Render the world from this player's perspective
        player_view = render(env, i)
        obs = fill(0.2f0, view_size, view_size, 3)
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
        obs[x_dst_range, y_dst_range, :] .= view(player_view, x_src_range, y_src_range, :)
        # add dim 1 to make it a 4D array for batch size
        obs = reshape(obs, view_size, view_size, 3, 1)
        observations[i] = obs
    end
    observations
end

function get_actions(observations, phenotypes::Vector{P}) where P<:AbstractPhenotype
    [pheno(obs) for (obs, pheno) in zip(observations, phenotypes)]
end

function get_player_color(viewing_player::Int, player_idx::Int)
    viewing_player == player_idx ? SELF_COLOR : OTHER_COLOR
end

# Helper function to get resource colors based on perspective
function get_resource_colors(perspective::Int)
    if perspective % 2 == 1
        return (apples = [1f0, 0f0, 0f0], bananas = [0f0, 1f0, 0f0])  # Red apples, Green bananas
    else
        return (apples = [0f0, 1f0, 0f0], bananas = [1f0, 0f0, 0f0])  # Green apples, Red bananas
    end
end

function render(env::TradeGridWorld, perspective::Int=1)
    n = env.n
    img = zeros(Float32, n, n, 3)
    
    # Render water pool in center
    center = n ÷ 2
    for x in 1:n, y in 1:n
        dx = x - center
        dy = y - center
        if sqrt(dx^2 + dy^2) <= env.pool_radius
            img[x, y, :] .= POOL_COLOR
        end
    end

    for idx in eachindex(env.players)
        player = env.players[idx]
        x_center = player.position[1] + 1  # Adjust for 1-based indexing
        y_center = player.position[2] + 1  # Adjust for 1-based indexing
        radius = PLAYER_RADIUS  # Circle radius

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
                    img[x, y, :] .= get_player_color(perspective, idx)
                end
            end
        end
    end

    # Get the correct colors for this perspective
    resource_colors = get_resource_colors(perspective)
    
    # Render apples and bananas on the grid, if there is anything, it starts at 0.5
    for x in 1:n, y in 1:n
        if env.grid_apples[x, y] > 0
            img[x, y, :] .= resource_colors.apples .* (0.5f0 + env.grid_apples[x,y]/(2*STARTING_RESOURCES))
        elseif env.grid_bananas[x, y] > 0
            img[x, y, :] .= resource_colors.bananas .* (0.5f0 + env.grid_bananas[x,y]/(2*STARTING_RESOURCES))
        end
    end
    img
end

function reset_map!(env::TradeGridWorld)
    # Clear grids
    fill!(env.grid_apples, 0.0)
    fill!(env.grid_bananas, 0.0)
    
    # Respawn resources in corners
    env.grid_apples[2, 2] = STARTING_RESOURCES
    env.grid_bananas[env.n-2, env.n-2] = STARTING_RESOURCES
    
    # Reset player resources
    for player in env.players
        player.resource_apples = 0.0
        player.resource_bananas = 0.0
    end
end

function save_gif(frames::Vector{Array{Float32,3}}, filename::String)
    rgb_frames = [permutedims(frame, (3, 1, 2)) for frame in frames]
    rgb_frames = [Array(colorview(RGB, frame)) for frame in rgb_frames]
    rgb_frames = cat(rgb_frames..., dims=3)
    FileIO.save(filename, rgb_frames)
end
