"""This script is used to play the environment interactively."""
from pddlgym_utils import play_env

if __name__ == "__main__":
    env_name = "PDDLEnvEasyblocks-v0"
    play_env(env_name, max_steps=100, step_time=0.1, mode="interactive", render=True, save_gif=True)