from gym.envs.registration import register

register(
    id="sorting-v0",
    entry_point="gym_sorting.envs:Sorting_Env",
    max_episode_steps=2000,
    kwargs={'max_steps_per_episode': 100, 'render':True, 'num_boxes':2, 'if_vision':False}
)
