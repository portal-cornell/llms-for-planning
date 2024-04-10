"""
This script is used to play the Sim2D environment interactively.

To run this script on an example, run the following command in the terminal:
    python play_script.py \
        --max_steps 10 \
        --mode random \
        --render_mode rgb_array \
        --fps 4 \
        --gif_file images/blocks_operator_actions.gif
"""
import argparse
from sim2d_utils import play_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the environment interactively")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum number of steps to simulate")
    parser.add_argument("--mode", type=str, choices=["random", "interactive"], help="Mode to use to select the action")
    parser.add_argument("--render_mode", type=str, choices=["human", "rgb_array"], help="Mode to use to render the environment")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second to render the GIF at")
    parser.add_argument("--gif_file", type=str, default=None, help="The name of the file to save the gif to.")
    args = parser.parse_args()

    play_env(args.max_steps, args.mode, args.render_mode, args.fps, args.gif_file)