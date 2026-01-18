from gym.envs.registration import register

register(
    id="avoiding-v0",
    entry_point="gym_avoiding.envs:ObstacleAvoidanceEnv",
    max_episode_steps=150,
)
