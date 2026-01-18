from gym.envs.registration import register

register(
    id="gate_insertion-v0",
    entry_point="gym_inserting.envs:Gate_Insertion_Env",
    max_episode_steps=2500,
)