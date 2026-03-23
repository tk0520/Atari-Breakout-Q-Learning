from gymnasium.wrappers import RecordVideo, AtariPreprocessing, FrameStackObservation
import gymnasium as gym
import numpy as np
import envpool
import ale_py
import random
import torch
import time

from agent.agent import CartPoleAgent, AtariAgent
from utils import train, train_with_trajectory
from recorder.setup import setup_env, setup_video_record
import settings

NUM_ENV = settings.ENVS_NUM
BATCH_SIZE = settings.ENVS_BATCH

train_envs = envpool.make(
    "Breakout-v5", env_type="gym", num_envs=NUM_ENV, batch_size=BATCH_SIZE, num_threads=6,
    stack_num=4, frame_skip=4, noop_max=30, gray_scale=True, seed=0, use_fire_reset=True
)
eval_env = setup_env()
eval_env = setup_video_record(eval_env)

agent = AtariAgent(BATCH_SIZE, train_envs.observation_space.shape[0], train_envs.action_space.n)
train(train_envs, eval_env, agent)
