import argparse
import os

from recorder import *

parser = parser = argparse.ArgumentParser()
parser.add_argument("model_file", type=str)
parser.add_argument("--model_dir", type=str, default="/home/solo/Desktop/paper-implementation/models", help="Model File")
parser.add_argument("--episode", type=str, default='', help="The number of episodes the chosen model has completed")

def main():
    ARGS = parser.parse_args()
    MODEL_PATH = os.path.join(ARGS.model_dir, ARGS.model_file)

    eval_env = setup_env()
    eval_env = setup_video_record(eval_env, ARGS.model_file)

    eval_agent = setup_agent(eval_env, MODEL_PATH)
    run(eval_env, eval_agent)

if __name__ == "__main__":
    main()