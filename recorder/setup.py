from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from gymnasium.wrappers import RecordVideo, Autoreset
import gymnasium as gym
import numpy as np
import argparse
import ale_py
import os

from .agent import EvaluationAgent

def setup_env():
    gym.register_envs(ale_py)

    RENDER_MODE = "rgb_array"
 
    eval_env = gym.make("ALE/Breakout-v5", render_mode=RENDER_MODE)
    eval_env = AtariPreprocessing(
        eval_env, 
        screen_size=84,       
        grayscale_obs=True,     
        frame_skip=1,               
        noop_max=0
    )
    eval_env = FrameStackObservation(eval_env, stack_size=4)
    return eval_env

def setup_video_record(eval_env):
    eval_env = RecordVideo(
        eval_env,
        video_folder="/home/solo/Desktop/paper-implementation/videos",
        episode_trigger=lambda x: True,
        name_prefix=f"breakout-DQN"
    )
    return eval_env

def setup_agent(eval_env, model_path):
    return EvaluationAgent(model_path, eval_env.observation_space.shape[0], eval_env.action_space.n)

def run(eval_env, eval_agent):
    observation, info = eval_env.reset()
    lives = info["lives"]
    done = False

    print("=" * 100)
    print("Info:", info)
    print("Evaluation Env Start...!")

    observation, reward, terminated, truncated, info = eval_env.step(1)

    while not done:
        action = eval_agent.act(observation)
        observation, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

        current_lives = info["lives"]
        if not lives == current_lives:
            observation, reward, terminated, truncated, info = eval_env.step(1)
            lives = current_lives
            print("Lives:", lives)
            continue
            
    print("Evaluation Env End...!")
    print("=" * 100)
    eval_env.close()