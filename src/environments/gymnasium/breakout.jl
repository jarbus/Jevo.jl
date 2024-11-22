export BreakoutV5
using Images, FileIO
using Random
using PythonCall

global gym, breakout_env, record_breakout_env, np

mutable struct BreakoutV5 <: AbstractEnvironment
    n_envs::Int
    n_steps::Int
    n_stack::Int
    record_prefix::String  
    env
    prev_obs
    step::Int
    done::Int
    reward::Float32
    prev_rewards
    gym
    np
end

function BreakoutV5(n_envs::Int, n_steps::Int, n_stack::Int, record_prefix::String="")
    return BreakoutV5(n_envs, n_steps, n_stack, record_prefix, nothing, nothing, 1, 0, 0, Float64[], nothing, nothing)
end

function done(env::BreakoutV5)
    d = env.done == env.n_envs
    d && !isempty(env.record_prefix) && env.env.close()
    d
end


function get_breakout_imports(env::BreakoutV5)
    if !isdefined(Jevo, :breakout_env)
        gym = pyimport("gymnasium")
        ale_py = pyimport("ale_py")
        print(gym.register_envs(ale_py))
        _env = gym.make("ALE/Breakout-v5")
        _env = gym.wrappers.ResizeObservation(_env, (84, 84))
        _env = gym.wrappers.TimeLimit(_env, max_episode_steps=pyint(env.n_steps))
        _env = gym.wrappers.FrameStackObservation(_env, stack_size=pyint(env.n_stack))
        Jevo.breakout_env = _env
        Jevo.gym = gym
        Jevo.np = pyimport("numpy")
    end
    if !isempty(env.record_prefix)
        if !isdefined(Jevo, :record_breakout_env)
            @info "Creating Breakout environment for recording"
            gym = Jevo.gym
            _env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
            _env = gym.wrappers.ResizeObservation(_env, (84, 84))
            _env = gym.wrappers.TimeLimit(_env, max_episode_steps=pyint(env.n_steps))
            _env = gym.wrappers.FrameStackObservation(_env, stack_size=pyint(env.n_stack))
            _env = gym.wrappers.RecordVideo(_env, pystr("video/"), name_prefix=pystr(env.record_prefix * "-$(time())"),  episode_trigger=@pyeval("lambda x: True"))
            Jevo.record_breakout_env = _env
        end
        _env = Jevo.record_breakout_env
    else
        _env = Jevo.breakout_env
    end
    Jevo.gym, _env, Jevo.np
end

function step!(env::BreakoutV5, ids::Vector{Int}, phenotypes::Vector)
    if isnothing(env.env)
        env.gym, env.env, env.np = get_breakout_imports(env)
        obs, info = env.env.reset()
        env.prev_obs = PyArray(obs)
    end

    obs = env.prev_obs
    obs = obs ./ 255f0
    frames = [obs[i, :, :, :] for i in 1:env.n_stack]
    obs = cat(frames..., dims=3)
    obs = reshape(obs, 84, 84, 3*env.n_stack, 1)

    action = phenotypes[1].chain(obs)[:, 1] |> argmax
    action = env.np.array(action .- 1).T

    obs, reward, terminated, truncated, info = env.env.step(action)
    env.step += 1
    reward = pyconvert(Float32, reward)
    push!(env.prev_rewards, reward)

    if Bool(terminated) || Bool(truncated)# || 
        #length(env.prev_rewards) > 20 && all(env.prev_rewards[end-20:end] .< 0)
        env.done += 1
        env.step = 1
        env.prev_rewards = Float64[]
        env.prev_obs = nothing
        return [Interaction(ids[1], [], reward)]
    end
    env.prev_obs = PyArray(obs)
    [Interaction(ids[1], [], reward)]
end
