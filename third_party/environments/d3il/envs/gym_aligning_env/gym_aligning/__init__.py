from gym.envs.registration import register

register(
    id="aligning-v0",
    entry_point="gym_aligning.envs:Robot_Push_Env",
    max_episode_steps=400,
    kwargs={'render':False, 'if_vision':False}
)
