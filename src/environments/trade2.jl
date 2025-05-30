using Images
using Images.ImageCore: colorview, RGB

export TradeGridWorld, render, LogTradeRatios, ClearTradeRatios 

# Helper functions for visualization and logging
function log_heatmap(matrix, perm, title, xlabel, ylabel, filename)
    sorted_matrix = matrix[perm, perm]
    heatmap(sorted_matrix, aspect_ratio=1, title=title, xlabel=xlabel, ylabel=ylabel)
    savefig(filename)
end

function log_scatter(x_values, y_values, title, xlabel, ylabel, filename)
    scatter(x_values, y_values, legend=false, title=title, xlabel=xlabel, ylabel=ylabel)
    savefig(filename)
end

function log_measurement(measurement, h5=true)
    @info(measurement)
    h5 && @h5(measurement)
end

const SELF_COLOR = [0.2f0, 0.2f0, 0.2f0]
const OTHER_COLOR = [0.2f0, 0.2f0, 0.2f0]
const PLAYER_RADIUS = 3
const STARTING_RESOURCES = 10f0
const POOL_REWARD = 1.0f0  # Reward for standing in water pool
const POOL_COLOR = [0.0f0, 0.0f0, 1.0f0]  # Blue color for water
const FOOD_BONUS_EPSILON = 0.1
const APPLE_COLOR = [1.0f0, 0.0f0, 0.0f0]  # Red color for apples
const BANANA_COLOR = [0.0f0, 1.0f0, 0.0f0]  # Green color for bananas

struct TradeRatio <: AbstractMetric end
struct NumApples <: AbstractMetric end
struct NumBananas <: AbstractMetric end
struct EnvMinResource <: AbstractMetric end
struct EnvSecondMinResource <: AbstractMetric end
struct PlayerMinResource <: AbstractMetric end
struct PlayerMaxResource <: AbstractMetric end


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

struct EnvMinResourceInteraction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    min_resource::Float64
end

struct EnvSecondMinResourceInteraction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    second_min_resource::Float64
end

struct PlayerMinResourceInteraction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    min_resource::Float64
end

struct PlayerMaxResourceInteraction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    max_resource::Float64
end


struct NumApplesInteraction <: AbstractInteraction
    individual_id::Int
    other_ids::Vector{Int}
    count::Float64
end

struct NumBananasInteraction <: AbstractInteraction
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
    render_filename::String # directory
    frames::Vector{Array{Float32,3}}
    perspective_frames::Vector  # each player's obs 
end

function TradeGridWorld(n::Int, p::Int, max_steps::Int=100, view_radius::Int=30, reset_interval::Int=25, pool_radius::Int=5, render_filename::String="")
    @assert max_steps % reset_interval == 0
    grid_apples = zeros(n, n)
    grid_bananas = zeros(n, n)
    players = PlayerState[]
    for i in 1:p
        position = Float64.( (rand(6:(n-6)), rand(6:(n-6))) )
        while too_close_to_others(position, i, players)
            position = Float64.( (rand(6:(n-6)), rand(6:(n-6))) )
        end
        push!(players, PlayerState(i, position, 0.0, 0.0))
    end

    env = TradeGridWorld(n, p, grid_apples, grid_bananas, players, 1, max_steps, view_radius, reset_interval, pool_radius, render_filename, Array{Float32, 3}[], [Array{Float32, 3}[] for i in 1:p])
    reset_map!(env)
    env
end

function log_trade_ratio(state, pops, h5)
    @assert length(pops) == 1 && length(pops[1]) == 1
    # extract all trade ratio interactions
    individuals = pops[1][1].individuals
    pop_ratios, pop_apples, pop_bananas = Float64[], Float64[], Float64[]
    pop_env_mins, pop_env_second_mins = Float64[], Float64[]
    pop_player_mins, pop_player_maxs = Float64[], Float64[]
    
    idx_map = Dict(ind.id=>idx for (idx, ind) in enumerate(individuals))
    distances = compute_pairwise_distances!(get_tree(pops[1][1]), idx_map |> keys |> collect |> Set)[2]
    env_min_matrix = zeros(Float32, length(idx_map), length(idx_map))
    env_second_min_matrix = zeros(Float32, length(idx_map), length(idx_map))
    player_min_matrix = zeros(Float32, length(idx_map), length(idx_map))
    player_max_matrix = zeros(Float32, length(idx_map), length(idx_map))
    trade_matrix_updated = fill(false, length(idx_map), length(idx_map))
    distance_matrix = fill(-1f0, length(idx_map), length(idx_map))
    
    for ind in individuals
        isempty(ind.interactions) && continue
        ind_ratios, ind_apples, ind_bananas = Float64[], Float64[], Float64[]
        ind_env_mins, ind_env_second_mins = Float64[], Float64[]
        ind_player_mins, ind_player_maxs = Float64[], Float64[]
        
        for int in ind.interactions
            id1, id2 = int.individual_id, int.other_ids[1]
            idx1, idx2 = idx_map[id1], idx_map[id2]

            min_max_ids = min(id1, id2), max(id1, id2)
            if min_max_ids in keys(distances)
                distance_matrix[idx1,idx2] = distances[min_max_ids]
            end

            if int isa TradeRatioInteraction
                push!(ind_ratios, int.trade_ratio)
            elseif int isa NumApplesInteraction
                push!(ind_apples, int.count)
            elseif int isa NumBananasInteraction
                push!(ind_bananas, int.count)
            elseif int isa EnvMinResourceInteraction
                push!(ind_env_mins, int.min_resource)
                trade_matrix_updated[idx1, idx2] = true
                env_min_matrix[idx1,idx2] = max(env_min_matrix[idx1,idx2], int.min_resource)
            elseif int isa EnvSecondMinResourceInteraction
                push!(ind_env_second_mins, int.second_min_resource)
                env_second_min_matrix[idx1,idx2] = max(env_second_min_matrix[idx1,idx2], int.second_min_resource)
            elseif int isa PlayerMinResourceInteraction
                push!(ind_player_mins, int.min_resource)
                player_min_matrix[idx1,idx2] = max(player_min_matrix[idx1,idx2], int.min_resource)
            elseif int isa PlayerMaxResourceInteraction
                push!(ind_player_maxs, int.max_resource)
                player_max_matrix[idx1,idx2] = max(player_max_matrix[idx1,idx2], int.max_resource)
            end
        end
        
        !isempty(ind_ratios) && push!(pop_ratios, mean(ind_ratios))
        !isempty(ind_apples) && push!(pop_apples, mean(ind_apples))
        !isempty(ind_bananas) && push!(pop_bananas, mean(ind_bananas))
        !isempty(ind_env_mins) && push!(pop_env_mins, maximum(ind_env_mins))
        !isempty(ind_env_second_mins) && push!(pop_env_second_mins, maximum(ind_env_second_mins))
        !isempty(ind_player_mins) && push!(pop_player_mins, maximum(ind_player_mins))
        !isempty(ind_player_maxs) && push!(pop_player_maxs, maximum(ind_player_maxs))
    end
    h5 && @assert all(trade_matrix_updated)
    # if any pop-level metrics are empty, set them to NaN
    isempty(pop_ratios) && push!(pop_ratios, NaN)
    isempty(pop_apples) && push!(pop_apples, NaN)
    isempty(pop_bananas) && push!(pop_bananas, NaN)
    isempty(pop_env_mins) && push!(pop_env_mins, NaN)
    isempty(pop_env_second_mins) && push!(pop_env_second_mins, NaN)
    isempty(pop_player_mins) && push!(pop_player_mins, NaN)
    isempty(pop_player_maxs) && push!(pop_player_maxs, NaN)

    ratio_m=StatisticalMeasurement(TradeRatio, pop_ratios, generation(state))
    apple_m=StatisticalMeasurement(NumApples, pop_apples, generation(state))
    banana_m=StatisticalMeasurement(NumBananas, pop_bananas, generation(state))
    env_min_m=StatisticalMeasurement(EnvMinResource, pop_env_mins, generation(state))
    env_second_min_m=StatisticalMeasurement(EnvSecondMinResource, pop_env_second_mins, generation(state))
    player_min_m=StatisticalMeasurement(PlayerMinResource, pop_player_mins, generation(state))
    player_max_m=StatisticalMeasurement(PlayerMaxResource, pop_player_maxs, generation(state))
    
    # Log all measurements using the helper function
    log_measurement(ratio_m, h5)
    log_measurement(apple_m, h5)
    log_measurement(banana_m, h5)
    log_measurement(env_min_m, h5)
    log_measurement(env_second_min_m, h5)
    log_measurement(player_min_m, h5)
    log_measurement(player_max_m, h5)

    if generation(state) % 100 == 0
        gen_padded = lpad(generation(state), 4, "0")
        
        # Environment min resource visualizations
        row_sums = vec(sum(env_min_matrix, dims=2))
        perm = sortperm(row_sums, rev=true)
        
        log_heatmap(
            env_min_matrix, perm,
            "Max EnvMinResource Matrix", "Individual", "Test",
            "media/max_env_minresource_matrix_$(gen_padded).png"
        )

        log_heatmap(
            env_second_min_matrix, perm,
            "Max EnvSecondMinResource Matrix", "Individual", "Test",
            "media/max_env_secondminresource_matrix_$(gen_padded).png"
        )
        
        # Calculate correlation between distance and env_min_matrix
        valid_indices = .!isnan.(vec(distance_matrix)) .& .!isnan.(vec(env_min_matrix)) .& (vec(distance_matrix) .>= 0)
        correlation = cor(vec(distance_matrix)[valid_indices], vec(env_min_matrix)[valid_indices])
        
        log_scatter(
            vec(distance_matrix), vec(env_min_matrix),
            "Distance vs Max EnvMinResource (r=$(round(correlation, digits=3)))", 
            "Phylogenetic Distance", "Max EnvMinResource",
            "media/distance_vs_env_minresource_$(gen_padded).png"
        )
        
        log_measurement(Measurement("DistanceEnvMinCorrelation", correlation, generation(state)), h5)

        # Environment second min resource visualizations
        # Calculate correlation between distance and env_second_min_matrix
        valid_indices = .!isnan.(vec(distance_matrix)) .& .!isnan.(vec(env_second_min_matrix)) .& (vec(distance_matrix) .>= 0)
        correlation = cor(vec(distance_matrix)[valid_indices], vec(env_second_min_matrix)[valid_indices])
        
        log_scatter(
            vec(distance_matrix), vec(env_second_min_matrix),
            "Distance vs Max EnvSecondMinResource (r=$(round(correlation, digits=3)))", 
            "Phylogenetic Distance", "Max EnvSecondMinResource",
            "media/distance_vs_env_secondminresource_$(gen_padded).png"
        )
        
        log_measurement(Measurement("DistanceEnvSecondMinCorrelation", correlation, generation(state)), h5)
        
        log_heatmap(
            player_min_matrix, perm,
            "Max PlayerMinResource Matrix", "Individual", "Test",
            "media/max_player_minresource_matrix_$(gen_padded).png"
        )
        
        # Calculate correlation between distance and player_min_matrix
        valid_indices = .!isnan.(vec(distance_matrix)) .& .!isnan.(vec(player_min_matrix)) .& (vec(distance_matrix) .>= 0)
        correlation = cor(vec(distance_matrix)[valid_indices], vec(player_min_matrix)[valid_indices])
        
        log_scatter(
            vec(distance_matrix), vec(player_min_matrix),
            "Distance vs Max PlayerMinResource (r=$(round(correlation, digits=3)))", 
            "Phylogenetic Distance", "Max PlayerMinResource",
            "media/distance_vs_player_minresource_$(gen_padded).png"
        )
        
        log_measurement(Measurement("DistancePlayerMinCorrelation", correlation, generation(state)), h5)
        
        log_heatmap(
            player_max_matrix, perm,
            "Max PlayerMaxResource Matrix", "Individual", "Test",
            "media/max_player_maxresource_matrix_$(gen_padded).png"
        )
        
        # Calculate correlation between distance and player_max_matrix
        valid_indices = .!isnan.(vec(distance_matrix)) .& .!isnan.(vec(player_max_matrix)) .& (vec(distance_matrix) .>= 0)
        correlation = cor(vec(distance_matrix)[valid_indices], vec(player_max_matrix)[valid_indices])
        
        log_scatter(
            vec(distance_matrix), vec(player_max_matrix),
            "Distance vs Max PlayerMaxResource (r=$(round(correlation, digits=3)))", 
            "Phylogenetic Distance", "Max PlayerMaxResource",
            "media/distance_vs_player_maxresource_$(gen_padded).png"
        )
        
        log_measurement(Measurement("DistancePlayerMaxCorrelation", correlation, generation(state)), h5)
    end

    individuals
end

isa_trade_interaction(int::AbstractInteraction) = int isa TradeRatioInteraction || 
                                                int isa NumApplesInteraction || 
                                                int isa NumBananasInteraction || 
                                                int isa EnvMinResourceInteraction || 
                                                int isa EnvSecondMinResourceInteraction ||
                                                int isa PlayerMinResourceInteraction ||
                                                int isa PlayerMaxResourceInteraction

function remove_trade_ratios!(state, individuals)
    for ind in individuals
        filter!(!isa_trade_interaction, ind.interactions)
    end
end

LogTradeRatios(;h5=true, kwargs...) = create_op("Reporter";
    retriever=PopulationRetriever(),
    operator=(s,ps)->log_trade_ratio(s,ps, h5),
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

function env_min_resource(players::Vector{PlayerState})
    min_resource = Inf
    for player in players
        min_resource = min(min_resource, player.resource_apples, player.resource_bananas)
    end
    return min_resource
end

function env_second_min_resource(players::Vector{PlayerState})
    values = Float64[]
    for player in players
        push!(values, player.resource_apples)
        push!(values, player.resource_bananas)
    end
    
    sort!(values)
    return values[2]  # Return the second smallest value
end

function player_min_resource(player::PlayerState)
    return min(player.resource_apples, player.resource_bananas)
end

function player_max_resource(player::PlayerState)
    return max(player.resource_apples, player.resource_bananas)
end

function collect_nearby_resources(grid::Array{Float64, 2}, player::PlayerState, amount)
    x, y = player.position
    x_min = max(1, ceil(Int, x - PLAYER_RADIUS))
    x_max = min(size(grid, 1), floor(Int, x + PLAYER_RADIUS))
    y_min = max(1, ceil(Int, y - PLAYER_RADIUS))
    y_max = min(size(grid, 2), floor(Int, y + PLAYER_RADIUS))

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

# Returns true if the position would be too close to any other player
function too_close_to_others(pos::Tuple{Float64,Float64}, current_player::Int, players::Vector{PlayerState})
    for (i, other) in enumerate(players)
        if i != current_player
            dx = pos[1] - other.position[1]
            dy = pos[2] - other.position[2]
            if sqrt(dx^2 + dy^2) < 2*(PLAYER_RADIUS)
                return true
            end
        end
    end
    return false
end

function step!(env::TradeGridWorld, ids::Vector{Int}, phenotypes::Vector{P}) where P<:AbstractPhenotype

    @assert length(ids) == length(phenotypes) == env.p
    
    interactions = []

    if env.step_counter == 1 && !isempty(env.render_filename)
        try
            mkpath(env.render_filename)
        catch
        end
        timestamp = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
        env.render_filename = joinpath(env.render_filename, "$(timestamp)_$(ids[1])_vs_$(ids[2])")
    end

    if env.step_counter > 1 && env.step_counter % env.reset_interval == 1
        append!(interactions, make_resource_interactions(env, ids))
        reset_map!(env)
    end

    if !isempty(env.render_filename) 
        push!(env.frames, render(env, 1))  # Always render from player 1's perspective
        #= for i in 1:env.p =#
        #=     # observations have a batch size associated with them, so we need to remove that =#
        #=     push!(env.perspective_frames[i], observations[i][:,:,:,1]) =#
        #= end =#

        render_txt = env.render_filename * ".txt"
        # write player states to txt file
        open_code = env.step_counter == 1 ? "w" : "a"
        open(render_txt, open_code) do f
            println(f, "Step: $(env.step_counter)")
            for player in env.players
                println(f, "Player $(player.id): $(round.(player.position, digits=2)), apples: $(round.(player.resource_apples, digits=2)), bananas: $(round.(player.resource_bananas, digits=2))")
            end
        end
    end

    for (i, id) in enumerate(ids)
        observations = make_observations(env, ids, phenotypes)
        other_id = ids[3-i]
        player = env.players[i]
        action_values = phenotypes[i](observations[i])
        @assert length(action_values) == 4
        # assert all actions are not nan or inf
        @assert !any(isnan.(action_values)) && !any(isinf.(action_values))
        dx, dy, place_action, pick_action = action_values
        place_action, pick_action = tanh(place_action), tanh(pick_action)
        movement_weight = abs(dx) + abs(dy)
        if movement_weight > 1
            dx = 2.5*dx / movement_weight
            dy = 2.5*dy / movement_weight
        end
        place_action = place_action * STARTING_RESOURCES
        pick_action = pick_action * STARTING_RESOURCES
        
        # Find closest valid position along movement vector
        prev_pos = player.position
        dx_unit = dx == 0 ? 0 : dx/sqrt(dx^2 + dy^2)
        dy_unit = dy == 0 ? 0 : dy/sqrt(dx^2 + dy^2)
        dist = sqrt(dx^2 + dy^2)
        
        # Try full movement first, then gradually reduce until valid
        test_dist = dist
        while test_dist > 0
            test_x = prev_pos[1] + dx_unit * test_dist
            test_y = prev_pos[2] + dy_unit * test_dist
            test_pos = (clamp(test_x, 1, env.n), clamp(test_y, 1, env.n))
            
            if !too_close_to_others(test_pos, i, env.players)
                player.position = test_pos
                break
            end
            test_dist -= 0.1
        end

        px = round(Int, player.position[1])
        py = round(Int, player.position[2])
        
        # Handle placing resources
        if place_action > 0  # Place primary resource
            amount = min(player.resource_apples, place_action)
            player.resource_apples -= amount
            env.grid_apples[px, py] += amount
        elseif place_action < 0  # Place secondary resource
            amount = min(player.resource_bananas, abs(place_action))
            player.resource_bananas -= amount
            env.grid_bananas[px, py] += amount
        end

        if pick_action > 0  # Pick primary resource
            player.resource_apples += collect_nearby_resources(env.grid_apples, player, pick_action)
        elseif pick_action < 0  # Pick secondary resource
            player.resource_bananas += collect_nearby_resources(env.grid_bananas, player, abs(pick_action))
        end

        # Check if player is in water pool
        center = env.n ÷ 2
        dx = player.position[1] - center
        dy = player.position[2] - center
        in_pool = sqrt(dx^2 + dy^2) <= env.pool_radius + PLAYER_RADIUS
        pool_bonus = in_pool ? POOL_REWARD : 0.0f0
        
        score = log(1.1+ 10player.resource_apples) * log(1.1+10player.resource_bananas) + pool_bonus
        push!(interactions, Interaction(id, [other_id], score))

        env.players[i] = player  # Update player state
    end
    env.step_counter += 1
    # Record trade ratio
    if done(env)
        if !isempty(env.render_filename)
            push!(env.frames, render(env, 1))  # Always render from player 1's perspective
            save_gif(env.frames, env.render_filename * ".gif")

            #= for i in 1:env.p =#
            #=     save_gif(env.perspective_frames[i], "$(root)_player_$(i).gif") =#
            #= end =#

        end
        append!(interactions, make_resource_interactions(env, ids))
    end
    return interactions
end

done(env::TradeGridWorld) = env.step_counter > env.max_steps

function make_resource_interactions(env::TradeGridWorld, ids::Vector{Int})
    interactions = []
    @assert env.p == 2
    for i in 1:env.p, j in (i+1):env.p

        # Environment-level metrics (across all players)
        env_min_resource_value = env_min_resource(env.players)
        env_second_min_resource_value = env_second_min_resource(env.players)

        push!(interactions, 
            EnvMinResourceInteraction(ids[i], [ids[j]], env_min_resource_value),
            EnvMinResourceInteraction(ids[j], [ids[i]], env_min_resource_value))

        push!(interactions, 
            EnvSecondMinResourceInteraction(ids[i], [ids[j]], env_second_min_resource_value),
            EnvSecondMinResourceInteraction(ids[j], [ids[i]], env_second_min_resource_value))
        
        # Player-level metrics (per player)
        for k in 1:env.p
            player_min = player_min_resource(env.players[k])
            player_max = player_max_resource(env.players[k])
            
            push!(interactions, 
                PlayerMinResourceInteraction(ids[k], [ids[k == 1 ? 2 : 1]], player_min))
            
            push!(interactions, 
                PlayerMaxResourceInteraction(ids[k], [ids[k == 1 ? 2 : 1]], player_max))
        end

        push!(interactions, 
            TradeRatioInteraction(ids[i], [ids[j]], inverse_absolute_difference(env.players[i])),
            TradeRatioInteraction(ids[j], [ids[i]], inverse_absolute_difference(env.players[j])))
        # compute primary
        push!(interactions, 
            NumApplesInteraction(ids[i], [ids[j]], env.players[i].resource_apples),
            NumBananasInteraction(ids[i], [ids[j]], env.players[i].resource_bananas))
        # compute secondary
        push!(interactions, 
            NumApplesInteraction(ids[j], [ids[i]], env.players[j].resource_apples),
            NumBananasInteraction(ids[j], [ids[i]], env.players[j].resource_bananas))
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
        px = round(Int, player.position[1])
        py = round(Int, player.position[2])
        
        # Calculate ranges for source and destination
        x_src_range = max(1, px - view_radius):min(env.n, px + view_radius)
        y_src_range = max(1, py - view_radius):min(env.n, py + view_radius)
        x_dst_start = view_radius + 1 - (px - first(x_src_range))
        y_dst_start = view_radius + 1 - (py - first(y_src_range))
        x_dst_range = x_dst_start:(x_dst_start + length(x_src_range)-1)
        y_dst_range = y_dst_start:(y_dst_start + length(y_src_range)-1)
        
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

function get_player_color(viewing_player::Int, player_idx::Int, players::Vector{PlayerState})
    color = viewing_player == player_idx ? copy(SELF_COLOR) : copy(OTHER_COLOR)
    increments = (1 .- color) ./ STARTING_RESOURCES  # apples increments are [1], banana increments are [2]
    color[1] += players[player_idx].resource_apples * increments[1]
    color[2] += players[player_idx].resource_bananas * increments[2]
    color
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

    # Iterate through all players and plot them, but do the current player's perspective last
    # to ensure they are on top of the others
    player_order = [i for i in 1:env.p if i != perspective]
    player_order = vcat(player_order, perspective)

    for idx in player_order
        player = env.players[idx]
        x_center = player.position[1]
        y_center = player.position[2]
        radius = PLAYER_RADIUS  # Circle radius

        # Determine the bounding box for the circle
        x_min = max(ceil(Int, x_center - radius), 1)
        x_max = min(floor(Int, x_center + radius), n)
        y_min = max(ceil(Int, y_center - radius), 1)
        y_max = min(floor(Int, y_center + radius), n)

        player_color = get_player_color(perspective, idx, env.players)

        for x in x_min:x_max
            for y in y_min:y_max
                # Compute the distance from the center
                dx = x - x_center
                dy = y - y_center
                distance = sqrt(dx^2 + dy^2)
                if distance <= radius
                    img[x, y, :] .= player_color
                end
            end
        end
    end

    # Render apples and bananas on the grid, if there is anything, it starts at 0.5
    for x in 1:n, y in 1:n
        # clear pixels to add food colors
        if env.grid_apples[x, y] > 0 || env.grid_bananas[x, y] > 0
            img[x, y, :] .= 0.0f0
        end
        if env.grid_apples[x, y] > 0
            img[x, y, :] += APPLE_COLOR .* (0.25f0 + env.grid_apples[x,y]/(1.4*STARTING_RESOURCES))
        end
        if env.grid_bananas[x, y] > 0
            img[x, y, :] += BANANA_COLOR .* (0.25f0 + env.grid_bananas[x,y]/(1.4*STARTING_RESOURCES))
        end
    end
    @assert all(0.0f0 .<= img .<= 1.0f0)
    img
end

function reset_map!(env::TradeGridWorld)
    # Clear grids
    fill!(env.grid_apples, 0.0)
    fill!(env.grid_bananas, 0.0)
    
    # Respawn resources in corners
    # env.grid_apples[2, 2] = STARTING_RESOURCES
    # env.grid_bananas[env.n-2, env.n-2] = STARTING_RESOURCES
    
    # Reset player resources
    for (idx, player) in enumerate(env.players)
        if idx == 1
            player.resource_apples = STARTING_RESOURCES
            player.resource_bananas = 0.0
        else
            player.resource_apples = 0.0
            player.resource_bananas = STARTING_RESOURCES
        end
    end
end

function save_gif(frames::Vector{Array{Float32,3}}, filename::String)
    rgb_frames = [permutedims(frame, (3, 1, 2)) for frame in frames]
    rgb_frames = [imresize(Array(colorview(RGB, frame)), (768,768)) for frame in rgb_frames]
    rgb_frames = cat(rgb_frames..., dims=3)
    FileIO.save(filename, rgb_frames)
end
