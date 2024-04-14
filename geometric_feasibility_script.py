"""
This script is used to run the geometric feasibility planner on an environment.

To run this script on an example, run the following command in the terminal:
    python geometric_feasibility_script.py \
        --experiment_name grocery_bot_plan \
        --prompt_description initial \
        --prompt_version 1.0.0 \
        --model gpt-4 \
        --temperature 0.7 \
        --seed 0 \
        --num_plans 1 \
        --beam_size 10 \
        --num_samples 10 \
        --gif_path ./planners/geometric_feasibility/images/image_plan.gif
"""
import argparse
import random

from prompt_builder.prompt_llm import prompt_llm
from planners.geometric_feasibility.v0_no_llm_scoring import plan
import planners.geometric_feasibility.sim2d_utils as sim2d_utils

from utils import fetch_messages

def generate_sim2d_plan(objs_to_put_away, locations, initial_state_of_fridge, preference, llm_params):
    """Generates a plan for the sim2d environment.
    
    Parameters:
        objs_to_put_away (List[str]):
            A list of object names to put away.
        locations (List[str]):
            A list of semantic locations where objects can be placed.
        initial_state_of_fridge (Dict[str, List[str]]):
            A dictionary mapping from location to a list of object names already in that location.
        preference (str):
            A string describing the user's preference for object placement.
        llm_params (Dict[str, Any]):
            A dictionary containing the parameters for the LLM model.
    
    Returns:
        text_plan (str):
            A string describing the plan to put away the objects.
    """
    user_prompt = f"""
    Objects: {objs_to_put_away}
    Locations: {locations}
    Initial State: {initial_state_of_fridge}
    Preference: "{preference}"
    """
    text_plan = prompt_llm(user_prompt, **llm_params)
    return text_plan

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

    # Prompt 1.0.1 test 1
    """
    objs_to_put_away = {
      "milk": {"width": 0.2, "height": 0.3},
      "chocolate milk": {"width": 0.25, "height": 0.3},
      "pineapple": {"width": 0.3, "height": 0.15},
      "cheddar": {"width": 0.15, "height": 0.1},
      "orange": {"width": 0.1, "height": 0.1},
      "pear": {"width": 0.15, "height": 0.2},
      "watermelon": {"width": 0.5, "height": 0.3},
    }
    locations = {
      "top shelf": {"x": 0, "y": 0.66, "width": 1, "height": 0.34},
      "middle shelf": {"x": 0, "y": 0.33, "width": 1, "height": 0.33},
      "bottom shelf": {"x": 0, "y": 0, "width": 1, "height": 0.33}
    }
    initial_state_of_fridge = {}
    preference = "I like putting yellow items on the middle shelf, milk on the top shelf, and fruit on the bottom shelf."
    """

    # Prompt 1.0.1 test 2
    """
    objs_to_put_away = {
        "apple": {"width": 0.1, "height": 0.1},
        "banana": {"width": 0.2, "height": 0.2},
        "cherries": {"width": 0.1, "height": 0.1},
        "chocolate_sauce": {"width": 0.125, "height": 0.25},
        "ketchup": {"width": 0.125, "height": 0.25},
        "oat_milk": {"width": 0.15, "height": 0.3},
        "whole_milk": {"width": 0.15, "height": 0.3},
        "mustard": {"width": 0.125, "height": 0.25},
        "potato": {"width": 0.2, "height": 0.1},
        "salad_dressing": {"width": 0.15, "height": 0.3},
        "tomato": {"width": 0.1, "height": 0.1}
    }
    locations = {
      "top shelf": {"x": 0, "y": 0.66, "width": 1, "height": 0.34},
      "middle shelf": {"x": 0, "y": 0.33, "width": 1, "height": 0.33},
      "bottom shelf": {"x": 0, "y": 0, "width": 1, "height": 0.33}
    }
    initial_state_of_fridge = {}
    preference = "Fruit on the left and vegetables on the right. I like my milk on the upper right and my condiments on the middle shelf."
    """

    # TODO(chalo2000): Move to Hydra config
    objs_to_put_away = ["apple", "banana", "cherries", "chocolate_sauce", "ketchup", "oat_milk", "whole_milk", "mustard", "potato", "salad_dressing", "tomato"]
    locations = ["top shelf", "left of top shelf", "right of top shelf", "middle shelf", "left of middle shelf", "right of middle shelf", "bottom shelf", "left of bottom shelf", "right of bottom shelf"]
    initial_state_of_fridge = {}
    preference = "Fruit on the left and vegetables on the right. I like my milk on the upper right and my condiments on the middle shelf."
    perception_values = None
    messages = fetch_messages(args.experiment_name, args.prompt_description, args.prompt_version)
    llm_params = {
        "messages": messages,
        "model": args.model,
        "temperature": args.temperature,
        "max_attempts": args.max_attempts,
        "sleep_time": args.sleep_time,
        "debug": args.debug
    }
    text_plan = generate_sim2d_plan(objs_to_put_away, locations, initial_state_of_fridge, preference, llm_params)
    print(f"LLM response:\n{text_plan}")

    env = sim2d_utils.make_sim2d_env(render_mode="rgb_array") # TODO(chalo2000): Allow setting environment to initial state
    # TODO(chalo2000): Make planner take in model which contains plan generation and skill extraction
    best_action_sequence = plan(env, args.num_plans, args.beam_size, args.num_samples, text_plans=[text_plan], perception_values=perception_values)
    if args.gif_path:
        sim2d_utils.save_replay(env, best_action_sequence, args.gif_path)

    