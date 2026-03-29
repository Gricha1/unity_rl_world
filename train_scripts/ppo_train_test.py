
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

time_int = int(time.time())

# Diretories for storing results 
log_dir = "results/big_square_obsts_builded_frame_stack_discrete_act_2/{}".format(time_int)
#log_dirTF = "stable_results/tensorflow_log_humanoid{}/".format(time_int) 
os.makedirs(log_dir, exist_ok=True)

env_name = "../train_without_obsts_1.x86_64.exe"

unity_env = UnityEnvironment(env_name, seed=1, side_channels=[channel])
channel.set_configuration_parameters(time_scale = 0.4)


env = UnityToGymWrapper(unity_env, uint8_visual=False)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    verbose=1,
    tensorboard_log=log_dir, device='cuda'
)

from stable_baselines3.common.callbacks import BaseCallback
import subprocess
import os

class EvalCallback(BaseCallback):
    def __init__(self, eval_freq, save_dir, eval_script):
        super().__init__()
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        self.eval_script = eval_script

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            path = os.path.join(self.save_dir, f"model_{self.num_timesteps}.zip")
            self.model.save(path)

            subprocess.Popen([
                "python",
                self.eval_script,
                path
            ])

        return True


print("start training!!!!!!!!!!!!!")
print("env render:", env.render(mode="rgb_array"))

callback = EvalCallback(
    eval_freq=10_000,
    save_dir=log_dir,
    eval_script="run_eval.py"
)

model.learn(
    total_timesteps=10_000_000,
    callback=callback
)