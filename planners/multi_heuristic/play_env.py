"""This script is used to play the environment interactively."""
import argparse
from pddlgym_utils import play_env

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env_name", required=True, help="The name of the environment.")
    args = argparser.parse_args()

    env_name = f"PDDLEnv{args.env_name.capitalize()}-v0"
    play_env(env_name, max_steps=100, step_time=0.1, mode="interactive", render=True, save_gif=True)