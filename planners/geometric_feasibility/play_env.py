"""This script is used to play the environment interactively."""
import argparse
from sim2d_utils import play_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the environment interactively")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to simulate")
    parser.add_argument("--mode", type=str, choices=["random", "interactive"], help="Mode to use to select the action")
    parser.add_argument("--render_mode", type=str, choices=["human", "rgb_array"], help="Mode to use to render the environment")
    parser.add_argument("--gif_path", type=str, default=None, help="Path to save the GIF to; doesn't save if None")
    args = parser.parse_args()

    play_env(args.max_steps, args.mode, args.render_mode, args.gif_path)