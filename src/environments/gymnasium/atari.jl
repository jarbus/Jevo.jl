export AtariEnv
using Images, FileIO
using Random
using PythonCall
using CUDA

global gym, atari_env, record_atari_env, np

mutable struct AtariEnv <: AbstractEnvironment
    name::String
    n_envs::Int
    n_steps::Int
    n_stack::Int
    record_prefix::String  
    env
    prev_obs
    step::Int
    done::Int
    reward::Float32
    skip_until::Int
    prev_rewards
    gym
    np
end

function AtariEnv(name::String, n_envs::Int, n_steps::Int, n_stack::Int, record_prefix::String="")
    return AtariEnv(name, n_envs, n_steps, n_stack, record_prefix, nothing, nothing, 1, 0, 0, rand(1:30), Float64[], nothing, nothing)
end

function done(env::AtariEnv)
    d = env.done == env.n_envs
    d && !isempty(env.record_prefix) && env.env.close()
    d
end


function get_atari_imports(env::AtariEnv)
    if !isdefined(Jevo, :atari_env)
        gym = pyimport("gymnasium")
        ale_py = pyimport("ale_py")
        gym.register_envs(ale_py)
        _env = gym.make(env.name)
        _env = gym.wrappers.ResizeObservation(_env, (84, 84))
        _env = gym.wrappers.FrameStackObservation(_env, stack_size=pyint(env.n_stack))
        Jevo.atari_env = _env
        Jevo.gym = gym
        Jevo.np = pyimport("numpy")
    end
    if !isempty(env.record_prefix)
        if !isdefined(Jevo, :record_atari_env)
            @info "Creating Atari environment for recording"
            gym = Jevo.gym
            _env = gym.make(env.name, render_mode="rgb_array")
            _env = gym.wrappers.ResizeObservation(_env, (84, 84))
            _env = gym.wrappers.FrameStackObservation(_env, stack_size=pyint(env.n_stack))
            _env = gym.wrappers.RecordVideo(_env, pystr("video/"), name_prefix=pystr(env.record_prefix * "-$(time())"),  episode_trigger=@pyeval("lambda x: True"))
            Jevo.record_atari_env = _env
        end
        _env = Jevo.record_atari_env
    else
        _env = Jevo.atari_env
    end
    Jevo.gym, _env, Jevo.np
end

function step!(env::AtariEnv, ids::Vector{Int}, phenotypes::Vector)
    if isnothing(env.env)
        env.gym, env.env, env.np = get_atari_imports(env)
        obs, info = env.env.reset()
        env.prev_obs = PyArray(obs) |> Array |> deepcopy
    end

    obs = env.prev_obs
    frames = [obs[i, :, :, :] for i in 1:env.n_stack]
    obs = cat(frames..., dims=3) |> gpu
    CUDA.synchronize()
    obs = obs ./ 255f0
    obs = reshape(obs, 84, 84, 3*env.n_stack, 1)

    if env.step < env.skip_until
        action = 1
    else
        CUDA.synchronize()
        action = phenotypes[1].chain(obs)
        CUDA.synchronize()
        action = cpu(action)
        action = argmax(action[:, 1])
    end
    action = env.np.array(action .- 1).T

    env.step += 1
    obs, reward, terminated, truncated, info = env.env.step(action)
    reward = pyconvert(Float32, reward)
    push!(env.prev_rewards, reward)

    if Bool(terminated) || Bool(truncated) || env.n_steps > 0 && env.step > env.n_steps 
        env.done += 1
        env.step = 1
        env.prev_rewards = Float64[]
        env.prev_obs = nothing
        return [Interaction(ids[1], [], reward)]
    end
    env.prev_obs = PyArray(obs) |> Array |> deepcopy
    [Interaction(ids[1], [], reward)]
end
