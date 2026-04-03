import envpool

from recorder.setup import setup_env, setup_video_record
from agent.agent import AtariAgent, AtariAgentPrioritizedReplay
from utils import train
import settings

NUM_ENV = settings.ENVS_NUM
BATCH_SIZE = settings.ENVS_BATCH

train_envs = envpool.make(
    "Breakout-v5", env_type="gym", num_envs=NUM_ENV, batch_size=BATCH_SIZE, num_threads=6,
    stack_num=4, frame_skip=4, noop_max=30, gray_scale=True, use_fire_reset=True, seed=49
)
eval_env = setup_env()
eval_env = setup_video_record(eval_env)

agent = AtariAgentPrioritizedReplay(BATCH_SIZE, train_envs.observation_space.shape[0], train_envs.action_space.n)
train(train_envs, eval_env, agent)
