using Images
using Images.ImageCore: colorview, RGB

export TradeGridWorld, render, LogTradeRatios, ClearTradeRatios 

mutable struct PlayerState
    id::Int
    position::Tuple{Float64, Float64}
    resource_apples::Float64
    resource_bananas::Float64
end

abstract type AbstractGridworld <: Jevo.AbstractEnvironment end


struct TradeRatioInteraction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    trade_ratio::Float64
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
    render_filename::String
    frames::Vector{Array{Float32,3}}
end

function TradeGridWorld(n::Int, p::Int, max_steps::Int=100, view_radius::Int=30, render_filename::String="")
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
    TradeGridWorld(n, p, grid_apples, grid_bananas, players, 1, max_steps, view_radius, render_filename, Array{Float32, 3}[])
end

struct TradeRatio <: AbstractMetric end

function log_trade_ratio(state, individuals)
    # extract all trade ratio interactions
    ratios = [int.trade_ratio for ind in individuals for int in ind.interactions if int isa TradeRatioInteraction]
    m=StatisticalMeasurement(TradeRatio, ratios, generation(state))
    @info(m)
    @h5(m)
    individuals
end

function remove_trade_ratios!(state, individuals)
    for ind in individuals
        filter!(interaction->!(interaction isa TradeRatioInteraction), ind.interactions)
    end
end

LogTradeRatios(;kwargs...) = create_op("Reporter";
    retriever=get_individuals,
    operator=log_trade_ratio,
    updater=remove_trade_ratios!,kwargs...)
ClearTradeRatios(;kwargs...) = create_op("Reporter";
    retriever=get_individuals,
    updater=remove_trade_ratios!,kwargs...)

# 0 when ratio is very unbalanced, 1 when ratio is even
function inverse_absolute_difference(apples, bananas)
    max_resource = max(apples, bananas)
    min_resource = min(apples, bananas)
    ratio =  max_resource / min_resource
    isnan(ratio) && return 0
    return 1 / (ratio + 1)
end
inverse_absolute_difference(player::PlayerState) = inverse_absolute_difference(player.resource_apples, player.resource_bananas)

function step!(env::TradeGridWorld, ids::Vector{Int}, phenotypes::Vector{P}) where P<:AbstractPhenotype

    !isempty(env.render_filename) && push!(env.frames, render(env, 1))  # Always render from player 1's perspective

    @assert length(ids) == length(phenotypes) == env.p
    interactions = []
    observations = make_observations(env, ids, phenotypes)
    actions = get_actions(observations, phenotypes)
    for (i, id) in enumerate(ids)
        player = env.players[i]
        action_values = actions[i]
        @assert length(action_values) == 4
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

        # Handle picking resources
        if pick_action > 0  # Pick primary resource
            if is_player_one && env.grid_apples[grid_x, grid_y] > 0
                amount = min(env.grid_apples[grid_x, grid_y], pick_action)
                player.resource_apples += amount
                env.grid_apples[grid_x, grid_y] -= amount
            elseif !is_player_one && env.grid_bananas[grid_x, grid_y] > 0
                amount = min(env.grid_bananas[grid_x, grid_y], pick_action)
                player.resource_bananas += amount
                env.grid_bananas[grid_x, grid_y] -= amount
            end
        elseif pick_action < 0  # Pick secondary resource
            if is_player_one && env.grid_bananas[grid_x, grid_y] > 0
                amount = min(env.grid_bananas[grid_x, grid_y], abs(pick_action))
                player.resource_bananas += amount
                env.grid_bananas[grid_x, grid_y] -= amount
            elseif !is_player_one && env.grid_apples[grid_x, grid_y] > 0
                amount = min(env.grid_apples[grid_x, grid_y], abs(pick_action))
                player.resource_apples += amount
                env.grid_apples[grid_x, grid_y] -= amount
            end
        end

        score = log(player.resource_apples + 1) + log(player.resource_bananas + 1)
        push!(interactions, Interaction(id, [], score))

        env.players[i] = player  # Update player state
    end
    env.step_counter += 1
    # Record trade ratio
    if done(env)
        for i in 1:env.p, j in (i+1):env.p
            push!(interactions, 
                TradeRatioInteraction(ids[i], [ids[j]], inverse_absolute_difference(env.players[i])),
                TradeRatioInteraction(ids[j], [ids[i]], inverse_absolute_difference(env.players[j])))
        end
        if !isempty(env.render_filename)
            push!(env.frames, render(env, 1))  # Always render from player 1's perspective
            # Save the frames as a gif
            rgb_frames = [permutedims(frame, (3, 1, 2)) for frame in env.frames]
            rgb_frames = [Array(colorview(RGB, frame)) for frame in rgb_frames]
            rgb_frames = cat(rgb_frames..., dims=3)
            FileIO.save(env.render_filename, rgb_frames)
        end
    end
    return interactions
end

done(env::TradeGridWorld) = env.step_counter > env.max_steps

# Creates RGB pixel observations for each player in the environment.
function make_observations(env::TradeGridWorld, ids::Vector{Int}, phenotypes::Vector{P}) where P<:AbstractPhenotype
    base_img = render(env)
    view_radius = env.view_radius
    view_size = 2 * view_radius + 1
    observations = Vector(undef, length(env.players))
    
    for (i, player) in enumerate(env.players)
        # Render the world from this player's perspective
        player_view = render(env, i)
        obs = ones(Float32, view_size, view_size, 3)
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

const SELF_COLOR = [0.67f0, 0.87f0, 0.73f0]
const OTHER_COLOR = [0.47f0, 0.60f0, 0.54f0]

function get_player_color(viewing_player::Int, player_idx::Int)
    viewing_player == player_idx ? SELF_COLOR : OTHER_COLOR
end

function render(env::TradeGridWorld, perspective::Int=1)
    n = env.n
    img = zeros(Float32, n, n, 3)

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
                    img[x, y, :] .= get_player_color(perspective, idx)
                end
            end
        end
    end
    # Helper function to get resource colors based on perspective
    function get_resource_colors(perspective::Int)
        if perspective % 2 == 1
            return (apples = [1f0, 0f0, 0f0], bananas = [0f0, 1f0, 0f0])  # Red apples, Green bananas
        else
            return (apples = [0f0, 1f0, 0f0], bananas = [1f0, 0f0, 0f0])  # Green apples, Red bananas
        end
    end

    # Get the correct colors for this perspective
    resource_colors = get_resource_colors(perspective)
    
    # Render apples and bananas on the grid
    for x in 1:n, y in 1:n
        if env.grid_apples[x, y] > 0
            img[x, y, :] .= resource_colors.apples .* (env.grid_apples[x,y]/10)
        elseif env.grid_bananas[x, y] > 0
            img[x, y, :] .= resource_colors.bananas .* (env.grid_bananas[x,y]/10)
        end
    end
    img
end
