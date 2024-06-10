"""
This script is used to run the geometric feasibility planner on an environment.

To run this script on an example, run the following command in the terminal:
    python geometric_feasibility_script.py \
        --experiment_name grocery_bot_plan \
        --prompt_description initial \
        --prompt_version 1.0.0 \
        --model gpt-4-turbo \
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

def generate_sim2d_plan(objs_to_put_away, locations, initial_state_of_fridge, preference, history, feedback, llm_params):
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
    # feedback_msg = "The previous plan was geometrically infeasible. The items that did not fit were ["melon", "apple"]."
    user_prompt = f"""
    Objects: {objs_to_put_away}
    Locations: {locations}
    Initial State: {initial_state_of_fridge}
    Preference: "{preference}"
    """
    feedback = feedback + '\n' if feedback is not None else ''
    user_prompt = f"{feedback}{user_prompt}" # Add feedback message to user prompt
    text_plan = prompt_llm(user_prompt, history=history, **llm_params)
    return user_prompt, text_plan

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
    
    # objs_to_put_away = {
    #   "milk": {"width": 0.2, "height": 0.3},
    #   "chocolate milk": {"width": 0.25, "height": 0.3},
    #   "pineapple": {"width": 0.3, "height": 0.15},
    #   "cheddar": {"width": 0.15, "height": 0.1},
    #   "orange": {"width": 0.1, "height": 0.1},
    #   "pear": {"width": 0.15, "height": 0.2},
    #   "watermelon": {"width": 0.5, "height": 0.3},
    # }
    # locations = {
    #   "top shelf": {"x": 0, "y": 0.66, "width": 1, "height": 0.34},
    #   "middle shelf": {"x": 0, "y": 0.33, "width": 1, "height": 0.33},
    #   "bottom shelf": {"x": 0, "y": 0, "width": 1, "height": 0.33}
    # }
    # initial_state_of_fridge = {}
    # preference = "I like putting yellow items on the middle shelf, milk on the top shelf, and fruit on the bottom shelf."
    
    """
    pickandplace("milk", {"x": 0.1, "y": 0.66})
    pickandplace("chocolate milk", {"x": 0.35, "y": 0.66})
    pickandplace("pineapple", {"x": 0.1, "y": 0.33})
    pickandplace("cheddar", {"x": 0.7, "y": 0.66})
    pickandplace("orange", {"x": 0.1, "y": 0})
    pickandplace("pear", {"x": 0.25, "y": 0})
    pickandplace("watermelon", {"x": 0.45, "y": 0})
    """

    # Prompt 1.0.1 test 2
    
    # objs_to_put_away = {
    #     "apple": {"width": 0.1, "height": 0.1},
    #     "banana": {"width": 0.2, "height": 0.2},
    #     "cherries": {"width": 0.1, "height": 0.1},
    #     "chocolate_sauce": {"width": 0.125, "height": 0.25},
    #     "ketchup": {"width": 0.125, "height": 0.25},
    #     "oat_milk": {"width": 0.15, "height": 0.3},
    #     "whole_milk": {"width": 0.15, "height": 0.3},
    #     "mustard": {"width": 0.125, "height": 0.25},
    #     "potato": {"width": 0.2, "height": 0.1},
    #     "salad_dressing": {"width": 0.15, "height": 0.3},
    #     "tomato": {"width": 0.1, "height": 0.1}
    # }
    # locations = {
    #   "top shelf": {"x": 0, "y": 0.66, "width": 1, "height": 0.34},
    #   "middle shelf": {"x": 0, "y": 0.33, "width": 1, "height": 0.33},
    #   "bottom shelf": {"x": 0, "y": 0, "width": 1, "height": 0.33}
    # }
    # initial_state_of_fridge = {}
    # preference = "Fruit on the left and vegetables on the right. I like my milk on the upper right and my condiments on the middle shelf."
    
    """
    pickandplace("apple", {"x": 0, "y": 0.66})
    pickandplace("banana", {"x": 0.1, "y": 0.66})
    pickandplace("cherries", {"x": 0.3, "y": 0.66})
    pickandplace("potato", {"x": 0.7, "y": 0.66})
    pickandplace("tomato", {"x": 0.9, "y": 0.66})

    pickandplace("oat_milk", {"x": 0.9, "y": 0.99})
    pickandplace("whole_milk", {"x": 0.7, "y": 0.99})

    pickandplace("ketchup", {"x": 0.1, "y": 0.33})
    pickandplace("mustard", {"x": 0.3, "y": 0.33})
    pickandplace("chocolate_sauce", {"x": 0.5, "y": 0.33})
    pickandplace("salad_dressing", {"x": 0.7, "y": 0.33})
    """

    """
    pickandplace("apple", {"x": 0.1, "y": 0.66})
    pickandplace("banana", {"x": 0.25, "y": 0.66})
    pickandplace("cherries", {"x": 0.5, "y": 0.66})
    pickandplace("potato", {"x": 0.8, "y": 0.66})
    pickandplace("tomato", {"x": 0.85, "y": 0.66})

    pickandplace("oat_milk", {"x": 0.7, "y": 0.66})
    pickandplace("whole_milk", {"x": 0.9, "y": 0.66})
    
    pickandplace("chocolate_sauce", {"x": 0.1, "y": 0.33})
    pickandplace("ketchup", {"x": 0.3, "y": 0.33})
    pickandplace("mustard", {"x": 0.5, "y": 0.33})
    pickandplace("salad_dressing", {"x": 0.7, "y": 0.33})
    """

    # TODO(chalo2000): Move to Hydra config
    # objs_to_put_away = ["apple", "banana", "cherries", "chocolate sauce", "ketchup", "oat milk", "whole milk", "mustard", "potato", "salad dressing", "onion"]
    # locations = ["top shelf", "left side of top shelf", "right side of top shelf", "middle shelf", "left side of middle shelf", "right side of middle shelf", "bottom shelf", "left side of bottom shelf", "right side of bottom shelf"]
    # initial_state_of_fridge = {}
    # preference = "Fruit on the left and vegetables on the right. I like my milk on the upper right and my condiments on the middle shelf."
    """
    Objects: ['mustard bottle', 'fanta can', 'apple juice bottle', 'pineapple', 'sprite can']
            Locations: ['top shelf', 'left side of top shelf', 'right side of top shelf', 'middle shelf', 'left side of middle shelf', 'right side of middle shelf', 'bottom shelf', 'left side of bottom shelf', 'right side of bottom shelf']
            Initial State: "{\"right side of middle shelf\": [\"soda orange fanta\"], \"left side of top shelf\": [\"tomato ketchup heinz\"], \"left side of middle shelf\": [\"cabbage\"], \"right side of bottom shelf\": [\"soda pepsi\"], \"right side of top shelf\": [\"coke bottle\", \"sprite soda green\"]}"
            Preference: "I want vegetables to be placed next to existing vegetables on the same shelf regardless of what shelf or what side it is. Fruits should be on the left side of the bottom shelf. I want drinks to be on the right side of the fridge. I want condiments to be placed next to existing condiments on the same shelf regardless of what shelf or what side it is."
    """
    objs_to_put_away = ["mustard", "fanta can", "apple juice", "pineapple", "sprite can"]
    locations = ["top shelf", "left side of top shelf", "right side of top shelf", "middle shelf", "left side of middle shelf", "right side of middle shelf", "bottom shelf", "left side of bottom shelf", "right side of bottom shelf"]
    initial_state_of_fridge = {
        "right side of middle shelf": ["fanta can"],
        "left side of top shelf": ["ketchup"],
        "left side of middle shelf": ["cabbage"],
        "right side of bottom shelf": ["pepsi can"],
        "right side of top shelf": ["coke bottle", "sprite can"]
    }
    preference = "I want vegetables to be placed next to existing vegetables on the same shelf regardless of what shelf or what side it is. Fruits should be on the left side of the bottom shelf. I want drinks to be on the right side of the fridge. I want condiments to be placed next to existing condiments on the same shelf regardless of what shelf or what side it is."
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
    history = []
    best_action_sequence = []
    best_obj_names = []
    feedback = None
    feedback_steps = 4
    i = 0
    while len(best_action_sequence) < len(objs_to_put_away) and i < feedback_steps:
        if i > 0:
            # Past first iteration, give feedback
            missing_objects = [obj for obj in objs_to_put_away if obj not in best_obj_names]
            feedback = f"The previous plan was geometrically infeasible. The items that did not fit were {missing_objects}."
        user_prompt, text_plan = generate_sim2d_plan(objs_to_put_away, locations, initial_state_of_fridge, preference, history, feedback, llm_params)
        print(f"LLM Prompt:\n{user_prompt}")
        history.append(user_prompt)
        print(f"LLM response:\n{text_plan}")
        history.append(text_plan)

        env = sim2d_utils.make_sim2d_env(render_mode="rgb_array") # TODO(chalo2000): Allow setting environment to initial state
        # TODO(chalo2000): Make planner take in model which contains plan generation and skill extraction
        best_action_sequence, best_obj_names = plan(env, args.num_plans, args.beam_size, args.num_samples, text_plans=[text_plan], perception_values=perception_values)
        # print(f"Best action sequence: {best_action_sequence}")
        # print(f"Length of best action sequence: {len(best_action_sequence)}")
        # print(f"Obj names: {best_obj_names}")
        i += 1
    if args.gif_path:
        sim2d_utils.save_replay(env, best_action_sequence, args.gif_path)

    