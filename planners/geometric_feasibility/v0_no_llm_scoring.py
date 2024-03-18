"""The V0 geometric feasibility planner uses an LLM to generate various plans for placing
objects in a fridge to satisfy a set of preferences. To gauge the geometric feasibility of
each plan, each plan is simulated in a fridge environment by sampling positions for each
object placement and scoring the outcome of each plan.

V0 aims to be quick by minimizing LLM usage when possible. Particularly, the LLM is only
queried to generate N plans and to convert placements such as "top shelf" into regions
to sample from. Scoring consists solely of computable heuristics like collision checking
and average packing space left. Other heuristics involving LLM queries like lookahead
potential may improve plan quality but are left to future versions to keep V0 fast.
"""
import argparse
from copy import deepcopy
import heapq
import numpy as np
import random

import sim2d_utils

PERCEPTION_CONSTANTS = {
    "shelf_heights": sim2d_utils.get_shelf_heights(),
    "objects": {
        "milk": {
            "width": 0.15,
            "height": 0.3,
            "color": (211, 219, 224)
        },
        "soda": {
            "width": 0.1,
            "height": 0.15,
            "color": (84, 54, 57)
        },
        "apple": {
            "width": 0.1,
            "height": 0.1,
            "color": (242, 58, 77)
        },
        "watermelon": {
            "width": 0.4,
            "height": 0.25,
            "color": (54, 125, 24)
        },
        "banana": {
            "width": 0.05,
            "height": 0.2,
            "color": (240, 235, 110)
        }
    }
}
def generate_plan():
    """Generates a plan for placing objects in the fridge.

    Returns:
        plan (list)
            A list of high-level language skills to execute in the environment.
    """
    obj_num = len(PERCEPTION_CONSTANTS["objects"])
    random_objects = random.sample(list(PERCEPTION_CONSTANTS["objects"].keys()), k=obj_num)
    random_locations = random.choices(list(PERCEPTION_CONSTANTS["shelf_heights"].keys()), k=obj_num)
    high_level_plan = []
    for i, (obj_name, loc_name) in enumerate(zip(random_objects, random_locations)):
        print(f"{i+1}) Place {obj_name} on the {loc_name} shelf.")
        object_dict = PERCEPTION_CONSTANTS["objects"][obj_name]
        object_bbox = (-1, -1, object_dict["width"], object_dict["height"])
        location_bbox = (0, PERCEPTION_CONSTANTS["shelf_heights"][loc_name], 1, 0.33)
        skill = ("pickandplace", (obj_name, object_bbox, location_bbox))
        high_level_plan.append(skill)
    return high_level_plan

def average_packing_space_left(env, sample, sample_locs):
    """Calculates the average packing space left in the environment.

    Parameters:
        env (gym.Env)
            The environment to calculate the average packing space left in.

    Returns:
        average_packing_space_left (float)
            The average packing space left in the environment.
    """
    _, l_y1, _, h, _ = sample # TODO: Clean up
    average_width = sum([obj["width"] for obj in PERCEPTION_CONSTANTS["objects"].values()]) 
    average_width /= len(PERCEPTION_CONSTANTS["objects"])
    score = 0
    curr_env = deepcopy(env)
    for sample_loc in sample_locs:
        action = (sample_loc, l_y1, average_width, h)
        collision = sim2d_utils.check_collision(curr_env, action)
        if not collision:
            curr_env.step(action)
            score += 1
    return score

def score_sample(env, sample, sample_locs):
    """Scores the outcome of a sample action in the environment.

    Parameters:
        env (gym.Env)
            The environment to score the sample in.
        sample (object)
            The sample action to score.
        sample_locs (list)
            The locations to sample from.

    Returns:
        score (float)
            The score of the sample.
    """
    # Collision score
    collision = sim2d_utils.check_collision(env, sample)
    if collision: return -float("inf")
    score = 1
    # Average packing space left score
    score += average_packing_space_left(env, sample, sample_locs)
    return score

def plan(env, num_plans, beam_size, num_samples):
    """Follows the V0 planning algorithm to output the best sequence of object placements.

    The V0 planning algorithm first generates N high-level language plans by few-shot prompting
    an LLM. The plan consists of a sequence of calls to lower-level skills; for V0 the only skill
    is `pickandplace(object, location)` where the object and location are in text. These texts
    are then converted into bounding boxes by querying an LLM with another few-shot prompt.

    Each plan is then converted into an action sequence of exact object placements. A beam
    of size B is maintained with an action sequence and its current score and for each skill
    call in the plan C candidates are sampled and scored. The beam is updated with the best
    B candidates until each skill has had object placements sampled. The final action sequence
    chosen is the best from the N plans generated.
    
    Parameters:
        env (gym.Env)
            The environment to plan in.
        num_plans (int)
            The number of plans (N) to generate.
        beam_size (int)
            The size of the beam (B) to maintain.
        num_samples (int)
            The number of samples (C) to take for each object placement.
    """
    # Generate N plans
    high_level_plans = []
    for i in range(num_plans): # TODO: Parallelize
        print(f"Plan {i+1}/{num_plans}")
        high_level_plan = generate_plan() # Generate correct plan
        high_level_plans.append(high_level_plan)
    
    # Generate best object placements for each plan
    best_action_sequences = []
    for high_level_plan in high_level_plans:
        # Maintain best candidates (score, action_sequence) on beam of size B
        beam = [(0, ())] * beam_size
        for skill in high_level_plan:
            skill_name, params = skill
            assert skill_name == "pickandplace"
            object_name, object_bbox, location_bbox = params
            _, _, o_w, o_h = object_bbox
            o_color = PERCEPTION_CONSTANTS["objects"][object_name]["color"]
            l_x1, l_y1, l_w, _ = location_bbox
            candidates = []
            for score, action_sequence in beam:
                # Simulate the current action sequence
                beam_env = deepcopy(env)
                beam_env.reset()
                for action in action_sequence:
                    beam_env.step(action)
                # Sample and score C candidate actions from each element on the beam
                sample_xs = list(np.linspace(l_x1, l_x1 + l_w, num_samples)) # Sample x
                for sample_x in sample_xs:
                    sample_action = (sample_x, l_y1, o_w, o_h, o_color) # Convert to action (x, y, w, h)
                    sample_score = score + score_sample(beam_env, sample_action, sample_xs) # Score
                    sample_action_sequence = action_sequence + (sample_action,)
                    candidates.append((sample_score, sample_action_sequence))
            # Update the beam with the best B candidates
            heapq.heapify(candidates)
            beam = heapq.nlargest(beam_size, set(candidates))
        best_action_sequences.append(max(beam))
    
    # Choose best object placement from all plans
    score, best_action_sequence = max(best_action_sequences)
                
    return best_action_sequence

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