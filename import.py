import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
environment_name = "CartPole-v1"
env = gym.make(environment_name, render_mode="human")  # или "rgb_array"
episodes = 50

for episode in range(1, episodes+1):
    state, _ = env.reset()  # reset() возвращает (obs, info)
    terminated = False
    truncated = False
    score = 0 
    
    while not terminated and not truncated:
        env.render()
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)  # 5 значений!
        score += reward
    print(f'Episode:{episode} Score:{score} Terminated{terminated} Truncated{truncated}')
env.close()