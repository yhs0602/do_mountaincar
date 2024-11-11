import sys
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

import wandb

from get_device import get_device

if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="mountain-car",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        sync_tensorboard=True,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 200000 == 0,
        video_length=400,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=get_device(0),
        tensorboard_log=f"runs/{run.id}",
    )
    model.learn(
        total_timesteps=9000000,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()
