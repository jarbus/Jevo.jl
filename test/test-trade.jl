using Jevo
using FileIO

using Jevo: Images
using Jevo.Images.ImageCore: colorview, RGB

mutable struct DummyPhenotype <: Jevo.AbstractPhenotype
    numbers::Vector{Float64}
end

function (p::DummyPhenotype)(x...)
    return p.numbers
end

view_radius = 30
@testset "Pickup/placedown" begin
    n = 10
    p = 2
    max_steps = 2
    env = TradeGridWorld(n, p, max_steps, view_radius, "pickup_placedown.gif")
    
    center_pos = (n/2, n/2)
    env.players[1].position = center_pos
    env.players[2].position = center_pos
    env.players[1].resource_apples = 10
    env.players[1].resource_bananas = 0
    env.players[2].resource_bananas = 10
    env.players[2].resource_apples = 0
    env.grid_apples .= 0f0
    env.grid_bananas .= 0f0
    player_moves = [ [[0,0,1,0],  [0,0,0,-1]],
                     [[0,0,1,0], [0,0,0,-1]]]
    
    while !done(env)
        ids = [player.id for player in env.players]
        phenotypes = [DummyPhenotype(player_moves[i][env.step_counter]) for i in eachindex(env.players)]
        step!(env, ids, phenotypes)
    end
    @test env.players[1].resource_apples == 9
    @test env.players[1].resource_bananas == 1
    @test env.players[2].resource_apples == 1
    @test env.players[2].resource_bananas == 9
end

@testset "record_player_perspective" begin
    n = 100
    p = 2
    max_steps = 50
    env = TradeGridWorld(n, p, max_steps, view_radius, "record_player_perspective.gif")

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
    n = 10
    p = 2
    max_steps = 3
    env = TradeGridWorld(n, p, max_steps, view_radius, "pickup_placedown.gif")
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
                     [[0,0,0,-1], [0,0,0,-1], [0,0,0,-1]]]
    
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
