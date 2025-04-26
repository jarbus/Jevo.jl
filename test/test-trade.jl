using Jevo
using FileIO

const POOL_RADIUS = 3

using Jevo: Images
using Jevo.Images.ImageCore: colorview, RGB

mutable struct DummyPhenotype <: Jevo.AbstractPhenotype
    numbers::Vector{Float64}
end

function (p::DummyPhenotype)(x...)
    return p.numbers
end

view_radius = 30
@testset "TradeV2" begin
    @testset "Pickup/placedown" begin
        n = 20
        p = 2
        max_steps = 3
        reset_interval = 3
        env = TradeGridWorld(n, p, max_steps, view_radius, reset_interval, POOL_RADIUS, "test/render/pickup_placedown/")

        center_pos = (n/2, n/2)
        env.players[1].position = center_pos
        env.players[2].position = center_pos
        env.players[1].resource_apples = 10
        env.players[1].resource_bananas = 0
        env.players[2].resource_bananas = 10
        env.players[2].resource_apples = 0
        env.grid_apples .= 0f0
        env.grid_bananas .= 0f0
        player_moves = [ [[0,0,1,0], [0,0,0,-1], [0,0,0,0]],
                         [[0,0,-1,0], [0,0,0,1], [0,0,0,0]]]

        for _ in 1:2
            ids = [player.id for player in env.players]
            phenotypes = [DummyPhenotype(player_moves[i][env.step_counter]) for i in eachindex(env.players)]
            step!(env, ids, phenotypes)
        end
        @test env.players[1].resource_apples < 10
        @test env.players[1].resource_bananas > 0
        @test env.players[2].resource_apples > 0
        @test env.players[2].resource_bananas < 10 
    end

    @testset "record_player_perspective" begin
        n = 100
        p = 2
        max_steps = 50
        reset_interval = max_steps
        env = TradeGridWorld(n, p, max_steps, view_radius, reset_interval, POOL_RADIUS, "test/render/record_player_perspective")

        p1_frames = []
        p2_frames = []
        while !done(env)
            ids = [player.id for player in env.players]
            phenotypes = [DummyPhenotype(randn(4)) for i in eachindex(env.players)]
            obs = Jevo.make_observations(env, ids, phenotypes)
            push!(p1_frames, obs[1])
            push!(p2_frames, obs[2])
            step!(env, ids, phenotypes)
        end
    end

    @testset "two_consecutive_pickups" begin
        # player picks up twice 
        n = 20
        p = 2
        max_steps = 3
        reset_interval = 3
        env = TradeGridWorld(n, p, max_steps, view_radius, reset_interval, POOL_RADIUS, "test/render/two_consecutive_pickups/")
        env.grid_apples .= 0f0
        env.grid_bananas .= 0f0

        env.grid_apples[4,3] = 1
        env.grid_apples[4,6] = 1
        env.grid_bananas[6,3] = 1
        env.grid_bananas[6,6] = 1

        # confirm agents dont pick up fruit outside of their range
        env.grid_apples[1,1] = 1
        env.grid_apples[1,10] = 1
        env.grid_bananas[10,1] = 1
        env.grid_bananas[10,10] = 1

        env.players[1].resource_apples = 10
        env.players[1].resource_bananas = 0
        env.players[2].resource_bananas = 10
        env.players[2].resource_apples = 0

        env.players[1].position = (5,5)
        env.players[2].position = (5,5)
        player_moves = [ [[0,0,0,-1],  [0,0,0,-1], [0,0,0,-1]],
                         [[0,0,0,1], [0,0,0,1], [0,0,0,1]]]
        
        while !done(env)
            ids = [player.id for player in env.players]
            phenotypes = [DummyPhenotype(player_moves[i][env.step_counter]) for i in eachindex(env.players)]
            step!(env, ids, phenotypes)
        end
        @test env.players[1].resource_apples == 10
        @test env.players[1].resource_bananas == 2
        @test env.players[2].resource_apples == 2
        @test env.players[2].resource_bananas == 10
    end

    @testset "automatic map reset" begin
        n = 20
        p = 2
        max_steps = 4  # Run for 4 steps to see reset at step 2
        reset_interval = 2
        env = TradeGridWorld(n, p, max_steps, view_radius, reset_interval, POOL_RADIUS, "")
        
        # Give players some resources
        env.players[1].resource_apples = 5.0
        env.players[1].resource_bananas = 3.0
        env.players[2].resource_apples = 2.0
        env.players[2].resource_bananas = 4.0
        
        # Clear and modify grid
        env.grid_apples .= 0.0
        env.grid_bananas .= 0.0
        env.grid_apples[5,5] = 2.0
        env.grid_bananas[3,3] = 3.0

        # Run for 1 step (no reset should occur)
        ids = [player.id for player in env.players]
        phenotypes = [DummyPhenotype([0.0, 0.0, 0.0, 0.0]) for _ in 1:p]
        step!(env, ids, phenotypes)
        
        # Verify no reset occurred
        @test env.grid_apples[5,5] == 2.0
        @test env.grid_bananas[3,3] == 3.0
        @test env.players[1].resource_apples == 5.0
        
        # Run another step (reset should not occur until the start of step 3)
        step!(env, ids, phenotypes)
        @test env.grid_apples[5,5] == 2.0
        @test env.grid_bananas[3,3] == 3.0
        @test env.players[1].resource_apples == 5.0
        
        step!(env, ids, phenotypes)
        # Verify reset occurred
        @test sum(env.grid_apples) == 0
        @test sum(env.grid_bananas) == 0
        
        # Check player resources were reset
        for player in env.players
            @test player.resource_apples == Jevo.STARTING_RESOURCES || player.resource_bananas == Jevo.STARTING_RESOURCES
        end
    end

    @testset "pool reward" begin
        n = 20
        p = 2
        max_steps = 2
        reset_interval = 2
        env = TradeGridWorld(n, p, max_steps, view_radius, reset_interval, POOL_RADIUS, "")
        
        # Place both players in center (pool)
        center = n ÷ 2
        env.players[1].position = (center, center)
        env.players[2].position = (center, center)
        
        # Run one step with players in pool
        ids = [player.id for player in env.players]
        phenotypes = [DummyPhenotype([0.0, 0.0, 0.0, 0.0]) for _ in 1:p]
        interactions = step!(env, ids, phenotypes)
        
        # Check both players got pool bonus
        pool_score_1 = interactions[1].score
        pool_score_2 = interactions[2].score
        
        # Move players out of pool
        env.players[1].position = (1.0, 1.0)
        env.players[2].position = (1.0, 1.0)
        
        # Run another step with players outside pool
        interactions = step!(env, ids, phenotypes)
        

        # Check scores without pool bonus
        no_pool_score_1 = interactions[1].score
        no_pool_score_2 = interactions[2].score
        @test pool_score_1 > no_pool_score_1
        @test pool_score_2 > no_pool_score_2
    end
end

#= @testset "1k steps" begin =#
#=     n=100 =#
#=     p=2 =#
#=     max_steps=1_000 =#
#=     start_time = time() =#
#=     env = TradeGridWorld(n, p, max_steps, view_radius) =#
#=     while !done(env) =#
#=         ids = [player.id for player in env.players] =#
#=         # Generate random actions for each player =#
#=         phenotypes = [DummyPhenotype(randn(4)) for _ in env.players] =#
#=         step!(env, ids, phenotypes) =#
#=     end =#
#=     end_time = time() =#
#=     println("1k steps passed in $(end_time - start_time) seconds.") =#
#= end =#
#==#
#= function run_random_episode(;n::Int=10, p::Int=2, max_steps::Int=100, view_radius::Int=30, output_filename::String="episode.gif") =#
#=     env = TradeGridWorld(n, p, max_steps=max_steps, view_radius=view_radius, render_filename=output_filename) =#
#=     frames = [] =#
#=     while !done(env) =#
#=         ids = [player.id for player in env.players] =#
#=         # Generate random actions for each player =#
#=         phenotypes = [DummyPhenotype(randn(4)) for _ in env.players] =#
#=         step!(env, ids, phenotypes) =#
#=         push!(frames, Jevo.render(env)) =#
#=     end =#
#= end =#


#= run_random_episode(n=100, p=2, max_steps=10) =#
#= test_overlapping_players(n=100, max_steps=10) =#
