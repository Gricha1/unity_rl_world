
from stable_baselines3 import PPO, SAC
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
channel = EngineConfigurationChannel()
#from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import time,os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

import os
import random
import time
from dataclasses import dataclass

env_name = "../train_without_obsts_1.x86_64.exe"

unity_env = UnityEnvironment(env_name, seed=1, side_channels=[channel])
channel.set_configuration_parameters(time_scale = 0.4)
unity_gym_env = UnityToGymWrapper(unity_env, uint8_visual=False) # OpenAI gym interface created using UNITY

"""
#class UnityGymnasiumWrapper(Env):
#    def __init__(self, unity_env):
#        self.unity_env = UnityToGymWrapper(unity_env, uint8_visual=False)  # Или передавайте параметры UnityToGymWrapper
#        
#    def step(self, action):
#        return self.unity_env.step(action)
#        
#    def reset(self, **kwargs):
#        return self.unity_env.reset(**kwargs)
#        
#    def render(self):
#        return self.unity_env.render()
#        
#    @property
#    def action_space(self):
#        return self.unity_env.action_space
#        
#    @property
#    def observation_space(self):
#        return self.unity_env.observation_space

#gym_env = UnityGymnasiumWrapper(test_unity_env)
#env = gym.wrappers.RecordEpisodeStatistics(gym_env)
"""



time_int = int(time.time())

# Diretories for storing results 
log_dir = "results/big_square_obsts_builded_frame_stack_discrete_act_2/{}".format(time_int)
#log_dirTF = "stable_results/tensorflow_log_humanoid{}/".format(time_int) 
os.makedirs(log_dir, exist_ok=True)

#env = Monitor(env, log_dir, allow_early_resets=True)
num_envs = 20
env = DummyVecEnv([lambda: unity_gym_env for i in range(num_envs)])  # The algorithms require a vectorized environment to run

model = PPO(ActorCriticPolicy, env, verbose=1, tensorboard_log=log_dir, device='cuda')


model.learn(int(100_000)) # you can change the step size

#time_int2 = int(time.time())

#print('TIME TAKEN for training',time_int-time_int2)

# save the model
model.save("PPO_ice_four_friends")

# del model
#model = PPO.load("PPO_unity_humanoid")
# evaluate_policy()

# mean_reward, std_reward = evaluate_policy(model, model.get_env(),n_eval_episodes=10)

obs= env.reset()

# Test the agent for 1000 steps after training
for i in range(100):
    #action, states = model.predict(obs)
    action = env.action_space.sample().reshape(1, -1)
    obs, rewards, done, info = env.step(action)
    #env.render()







