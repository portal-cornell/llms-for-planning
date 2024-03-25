"""Run the geometric feasibility planner"""
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
    parser.add_argument("--gif_path", type=str, default=None, help="Path to save the GIF to; doesn't save if None")
    args = parser.parse_args()

    random.seed(args.seed)
    env = sim2d_utils.make_sim2d_env(render_mode="rgb_array")
    best_action_sequence = plan(env, args.num_plans, args.beam_size, args.num_samples)
    # print(best_action_sequence)
    if args.gif_path:
        sim2d_utils.save_replay(env, best_action_sequence, args.gif_path)