"""
This script is used to play the environment interactively.

To run this script on an example, run the following command in the terminal:
    python play_script.py \
        --env_name blocks_operator_actions \
        --max_steps 100 \
        --step_time 0.1 \
        --fps 4 \
        --gif_file images/blocks_operator_actions.gif
"""

import argparse
from pddlgym_utils import play_env

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--env_name", required=True, help="The name of the environment.")
    argparser.add_argument("--max_steps", type=int, default=100, help="The maximum number of steps to take in the environment.")
    argparser.add_argument("--step_time", type=float, default=0.1, help="The time to pause between steps.")
    argparser.add_argument("--fps", type=int, default=4, help="The frames per second to save the gif as.")
    argparser.add_argument("--gif_file", help="The name of the file to save the gif to.")
    # TODO(chalo2000): Change between environments (PDDLGym vs. Robotouille)
    args = argparser.parse_args()

    env_name = f"PDDLEnv{args.env_name.capitalize()}-v0"
    play_env(
        env_name,
        max_steps=args.max_steps,
        step_time=args.step_time,
        fps=args.fps,
        mode="interactive",
        render=True,
        gif_file=args.gif_file
    )