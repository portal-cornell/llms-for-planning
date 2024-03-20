import argparse
import random

from prompt_builder.prompt_llm import prompt_llm
from planners.geometric_feasibility.v0_no_llm_scoring import plan
import planners.geometric_feasibility.sim2d_utils as sim2d_utils


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment_name", required=True, help="The name of the experiment for the prompt.")
    argparser.add_argument("--prompt_description", required=True, help="The description of the prompt to test.")
    argparser.add_argument("--prompt_version", required=True, help="The version of the prompt to test.")
    argparser.add_argument("--model", default="gpt-3.5-turbo", help="The LLM model to query.")
    argparser.add_argument("--temperature", default=0.0, type=float, help="The LLM temperature.")
    argparser.add_argument("--max_attempts", default=10, type=int, help="The number of attempts to query the LLM before giving up")
    argparser.add_argument("--debug", action="store_true", help="Whether or not to mock an LLM response")
    argparser.add_argument("--sleep_time", default=5, type=int, help="The number of seconds to sleep after a failed query before requerying")
    argparser.add_argument("--seed", type=int, default=0, help="Random seed")
    argparser.add_argument("--num_plans", type=int, default=10, help="Number of plans to generate")
    argparser.add_argument("--beam_size", type=int, default=10, help="Size of the beam to maintain")
    argparser.add_argument("--num_samples", type=int, default=10, help="Number of samples to take for each object placement")
    argparser.add_argument("--gif_path", type=str, default=None, help="Path to save the GIF to; doesn't save if None")
    args = argparser.parse_args()

    random.seed(args.seed)
    # Objects: ["apple", "banana", "cherries", "chocolate_sauce", "ketchup", "lettuce", "almond_milk", "oat_milk", "whole_milk", "mustard", "onion", "orange", "pear", "potato", "salad_dressing", "tomato"]
    # Locations: ["top shelf", "middle shelf", "bottom shelf"]
    # Preference: ...
    user_prompt = """
    Objects: ["apple", "banana", "cherries", "chocolate_sauce", "ketchup", "oat_milk", "whole_milk", "mustard", "potato", "salad_dressing", "tomato"]
    Locations: ["top shelf", "middle shelf", "bottom shelf"]
    Preference: "Keep milk and chocolate together and then vegetables and condiments together"
    """
    text_plan = prompt_llm(
        user_prompt,
        args.experiment_name,
        args.prompt_description,
        args.prompt_version,
        args.model,
        args.temperature,
        max_attempts=args.max_attempts,
        sleep_time=args.sleep_time,
        debug=args.debug
    )
    print(f"LLM response:\n{text_plan}")

    env = sim2d_utils.make_sim2d_env(render_mode="rgb_array")
    best_action_sequence = plan(env, args.num_plans, args.beam_size, args.num_samples, text_plans=[text_plan])
    if args.gif_path:
        sim2d_utils.save_replay(env, best_action_sequence, args.gif_path)

    