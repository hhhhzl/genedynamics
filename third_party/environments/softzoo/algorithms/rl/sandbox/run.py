from stable_baselines3 import PPO

from . import env


model = PPO('MultiInputPolicy', 'Ground-Caterpillar-v0', verbose=2, device='cpu')
model.learn(total_timesteps=100)
