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
import heapq
import numpy as np
from copy import deepcopy

def generate_plan():
    pass

def score_sample(env, sample):
    pass

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
        env (gym.Env): The environment to plan in.
        num_plans (int): The number of plans (N) to generate.
        beam_size (int): The size of the beam (B) to maintain.
        num_samples (int): The number of samples (C) to take for each object placement.
    """
    # Generate N plans
    plans = []
    for _ in range(num_plans): # TODO: Parallelize
        plan = generate_plan() # Generate correct plan
        plans.append(plan)
    
    # Generate best object placements for each plan
    best_action_sequences = []
    for plan in plans:
        # Maintain best candidates (score, action_sequence) on beam of size B
        beam = [(0, [])] * beam_size
        for skill in plan:
            skill_name, params = skill
            assert skill_name == "pickandplace"
            object_bbox, location_bbox = params
            _, _, o_w, o_h = object_bbox
            l_x1, l_y1, l_w, _ = location_bbox
            candidates = []
            for score, action_sequence in beam:
                # Simulate the current action sequence
                beam_env = deepcopy(env)
                for action in action_sequence:
                    beam_env.step(action)
                # Sample and score C candidate actions from each element on the beam
                sample_xs = list(np.linspace(l_x1, l_x1 + l_w, num_samples)) # Sample x
                for sample_x in sample_xs:
                    sample_action = (sample_x, l_y1, o_w, o_h) # Convert to action (x, y, w, h)
                    sample_score = score + score_sample(beam_env, sample_action) # Score
                    sample_action_sequence = action_sequence + [sample_action]
                    candidates.append((sample_score, sample_action_sequence))
            # Update the beam with the best B candidates
            heapq.heapify(candidates)
            beam = heapq.nlargest(beam_size, set(candidates))
        best_action_sequences.append(max(beam))
    
    # Choose best object placement from all plans
    best_action_sequence = max(best_action_sequences)
                
    return best_action_sequence
