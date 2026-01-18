from gym.envs.registration import register

register(
    id="stacking-v0",
    entry_point="gym_stacking.envs:CubeStacking_Env",
    max_episode_steps=2000,
    kwargs={'max_steps_per_episode': 1000, 'render':True, 'if_vision':False}
)
