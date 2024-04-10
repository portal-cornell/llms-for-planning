"""
This script is used to run the geometric feasibility planner on the Sim2D environment.

Note this script is intended mainly for debugging purposes. To plan using the LLM, 
use the top-level script instead which imports the `prompt_builder` package.

To run this script on an example, run the following command in the terminal:
    python plan_script.py \
        --seed 1 \
        --num_plans 1 \
        --beam_size 2 \
        --num_samples 5 \
        --gif_path images/sim2d.gif
"""
import argparse
import random

import sim2d_utils
from v0_no_llm_scoring import plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V0 geometric feasibility planner")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_plans", type=int, default=10, help="Number of plans to generate")
    parser.add_argument("--beam_size", type=int, default=10, help="Size of the beam to maintain")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to take for each object placement")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second to render the GIF at")
    parser.add_argument("--gif_path", type=str, default=None, help="Path to save the GIF to; doesn't save if None")
    args = parser.parse_args()

    random.seed(args.seed)
    env = sim2d_utils.make_sim2d_env(render_mode="rgb_array")
    best_action_sequence = plan(env, args.num_plans, args.beam_size, args.num_samples)
    print(best_action_sequence)
    if args.gif_path:
        sim2d_utils.save_replay(env, best_action_sequence, args.gif_path, fps=args.fps)