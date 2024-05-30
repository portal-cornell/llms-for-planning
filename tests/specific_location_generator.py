import random
import json
from copy import deepcopy
from collections import OrderedDict

import sys
sys.path.append('../planners/geometric_feasibility/') # Hack to import plan from v0_no_llm_scoring
import v0_no_llm_scoring # Must uncomment `import sim2d_utils` in v0_no_llm_scoring.py to work
import sim2d_utils

import os
import contextlib

CATEGORY_DICT = {
	"fruits": ["apple", "banana", "cherries", "orange", "pear", "watermelon", "papaya", "avocado", "mango", "pomegranate", "honeydew"],
	"dairy products": ["almond milk", "oat milk", "whole milk", "cheese block", "yogurt", "butter", "egg", "sliced cheese", "ice cream", "string cheese"],
	"juice and soft drinks": ["fanta", "pepsi", "coke bottle", "ginger ale", "gatorade", "strawberry milk", "banana milk", "orange juice", "apple juice", "cranberry juice"],
	"condiments": ["chocolate sauce", "ketchup", "mustard", "salad dressing", "soy sauce", "hot sauce", "nutella", "maple syrup", "whipped cream", "peanut butter"],
	"vegetables": ["lettuce", "onion", "potato", "tomato", "eggplant", "pumpkin", "bell pepper", "carrot", "radish", "corn"]
}

def generate_ground_truth_preference():
    """Returns a ground truth preference dictionary and string.
    
    The ground truth preference should only contain specific locations for each object category.

    Example:
    Dict: {
        'fruits': 'right side of top shelf',
        'dairy products': 'right side of top shelf',
        'juice and soft drinks': 'left side of bottom shelf',
        'condiments': 'right side of top shelf',
        'vegetables': 'left side of middle shelf'
    }
    String: I want fruits on the right side of top shelf, dairy products on the right side of top shelf, juice and soft drinks on the left side of bottom shelf, condiments on the right side of top shelf, vegetables on the left side of middle shelf.

    Returns:
        gt_preference_dict (dict): The ground truth preference dictionary.
        gt_preference_str (str): The ground truth preference string.
    """
    gt_preference_dict = {}
    chosen_locations = []
    categories = list(CATEGORY_DICT.keys())
    gt_preference_str = "I want "
    i = 0
    while len(chosen_locations) < len(categories):
        direction = random.choice(["left", "right"])
        shelf = random.choice(["top", "middle", "bottom"])
        location = f"{direction} side of {shelf} shelf"
        if location in chosen_locations:
            continue
        chosen_locations.append(location)
        category = categories[i]
        gt_preference_dict[category] = location
        gt_preference_str += f"{category} on the {location}, "
        i += 1
    gt_preference_str = gt_preference_str[:-2] + "."
    return gt_preference_dict, gt_preference_str

def generate_demonstrations(env, gt_preference_dict, num_demonstrations=2, demo_objects=7, obj_to_place=4):
    """Returns a list of demonstrations for the specific location preference.
    
    Parameters:
        env (gym.Env): The environment to generate the demonstrations in.
        gt_preference_dict (dict): The ground truth preference dictionary.
        num_demonstrations (int): The number of demonstrations to generate.
        demo_objects (int): The number of objects to put away in each demonstration.
        obj_to_place (int): The number of objects to place in each demonstration.
    
    Returns:
        demonstrations (list): A list of demonstrations.
        viz_data (list): A list of visualization data for each demonstration.
    """
    demonstrations = []
    viz_data = []
    ordered_categories = list(CATEGORY_DICT.keys())
    ablated_locations = [random.choice(ordered_categories)] # Ensure one category always persists over demonstrations
    i = 0
    while i < num_demonstrations:
        # Generate 'demo_objects' objects to put away that encompass all categories
        num_categories = len(ordered_categories)
        extra_objects = demo_objects - num_categories
        demo_categories = ordered_categories + random.choices(ordered_categories, k=extra_objects)
        all_objects = [random.choice(CATEGORY_DICT[category]) for category in demo_categories]
        if len(set(all_objects)) != len(all_objects):
            # Regenerate the demonstration if there are duplicate objects
            continue
        
        # Generate the final state of the fridge
        text_plan = ""
        for object, category in zip(all_objects, demo_categories):
            # Construct text plan
            location = gt_preference_dict[category]
            text_skill = f'pickandplace("{object}", "{location}")'
            text_plan += text_skill + "\n"
        text_plan = text_plan[:-1] # Remove last newline character
        
        # Execute text plan
        env.reset()

        # Suppress print output
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                best_action_sequence, best_obj_names = v0_no_llm_scoring.plan(env, 1, 10, 10, text_plans=[text_plan], perception_values=None)

        if len(best_action_sequence) < len(all_objects):
            continue # Regenerate the demonstration if the plan is infeasible
        
        final_state = {}
        for action, obj_name in zip(best_action_sequence, best_obj_names):
            object_coord_pair = (obj_name, (action[0], action[1]))
            category = [category for category in ordered_categories if obj_name in CATEGORY_DICT[category]][0]
            location = gt_preference_dict[category]
            final_state[location] = final_state.get(location, ()) + (object_coord_pair,)
        
        # Ablate the final state to generate the initial state
        initial_state = deepcopy(final_state)
        actual_ablations = []
        ablations = random.sample(list(gt_preference_dict.values()), k=len(gt_preference_dict))
        for ablation in ablations:
            if ablation in ablated_locations:
                # Should not ablate the same category (corresponding to location) twice to ensure coverage
                print(f"[-] Ablation {ablation} already performed. Skipping.")
                continue
            temp_initial_state = initial_state.copy()
            temp_initial_state.pop(ablation)
            if sum([len(objs) for objs in temp_initial_state.values()]) < demo_objects - obj_to_place:
                # Skip ablation if it results in less than 'obj_to_place' objects
                print(f"[-] Ablation {ablation} resulted in {sum([len(objs) for objs in temp_initial_state.values()])} objects. Skipping.")
                continue
            initial_state.pop(ablation)
            actual_ablations.append(ablation)
        if sum([len(objs) for objs in initial_state.values()]) != demo_objects - obj_to_place:
            # Regenerate the demonstration if the ablation results in less than 'obj_to_place' objects
            print(f"[-] Ablation resulted in {sum([len(objs) for objs in initial_state.values()])} objects. Regenerating demonstration.")
            continue
        ablated_objects = set(initial_state.values()).symmetric_difference(set(final_state.values()))
        objects_to_put_away = [obj_loc[0] for objs in ablated_objects for obj_loc in objs]        
        demonstration = {
            "objects_to_put_away": objects_to_put_away,
            "initial_state": initial_state,
            "final_state": final_state
        }
        demonstrations.append(demonstration)
        viz_data.append((objects_to_put_away, best_action_sequence, best_obj_names))
        ablated_locations += actual_ablations
        i += 1
    return demonstrations, viz_data

def generate_scenarios(env, gt_preference_dict, num_scenarios=2, scenario_objects=10, obj_to_place=6):
    """Returns a list of scenarios for the specific location preference.
    
    Parameters:
        env (gym.Env): The environment to generate the scenarios in.
        gt_preference_dict (dict): The ground truth preference dictionary.
        num_scenarios (int): The number of scenarios to generate.
        scenario_objects (int): The number of objects to put away in each scenario.
        obj_to_place (int): The number of objects to place in each scenario.
    
    Returns:
        scenarios (list): A list of scenarios.
        viz_data (list): A list of visualization data for each scenario.
    """
    scenarios = []
    viz_data = []
    ordered_categories = list(CATEGORY_DICT.keys())
    i = 0
    while i < num_scenarios:
        # Generate 'scenario_objects' objects to put away that encompass all categories
        num_categories = len(ordered_categories)
        extra_objects = scenario_objects - num_categories
        scenario_categories = ordered_categories + random.choices(ordered_categories, k=extra_objects)
        all_objects = [random.choice(CATEGORY_DICT[category]) for category in scenario_categories]
        if len(set(all_objects)) != len(all_objects):
            # Regenerate the scenario if there are duplicate objects
            continue
        
        # Generate the final state of the fridge
        text_plan = ""
        for object, category in zip(all_objects, scenario_categories):
            # Construct text plan
            location = gt_preference_dict[category]
            text_skill = f'pickandplace("{object}", "{location}")'
            text_plan += text_skill + "\n"
        text_plan = text_plan[:-1] # Remove last newline character
        
        # Execute text plan
        env.reset()

        # Suppress print output
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                best_action_sequence, best_obj_names = v0_no_llm_scoring.plan(env, 1, 10, 10, text_plans=[text_plan], perception_values=None)

        if len(best_action_sequence) < len(all_objects):
            continue # Regenerate the scenario if the plan is infeasible
        
        final_state = {}
        for action, obj_name in zip(best_action_sequence, best_obj_names):
            object_coord_pair = (obj_name, (action[0], action[1]))
            category = [category for category in ordered_categories if obj_name in CATEGORY_DICT[category]][0]
            location = gt_preference_dict[category]
            final_state[location] = final_state.get(location, ()) + (object_coord_pair,)
        
        # Ablate the final state to generate the initial state
        initial_state = deepcopy(final_state)
        ablations = random.sample(list(gt_preference_dict.values()), k=len(gt_preference_dict))
        for ablation in ablations:
            temp_initial_state = initial_state.copy()
            ablate_amount = random.randint(1, len(temp_initial_state[ablation]))
            temp_initial_state[ablation] = temp_initial_state[ablation][ablate_amount:]
            if sum([len(objs) for objs in temp_initial_state.values()]) < scenario_objects - obj_to_place:
                # Skip ablation if it results in less than 'obj_to_place' objects
                print(f"[-] Ablation {ablation} resulted in {sum([len(objs) for objs in temp_initial_state.values()])} objects. Skipping.")
                continue
            initial_state[ablation] = initial_state[ablation][ablate_amount:]
        if sum([len(objs) for objs in initial_state.values()]) != scenario_objects - obj_to_place:
            # Regenerate the demonstration if the ablation results in less than 'obj_to_place' objects
            print(f"[-] Ablation resulted in {sum([len(objs) for objs in initial_state.values()])} objects. Regenerating demonstration.")
            continue
        initial_state_objs = [obj_loc[0] for objs in initial_state.values() for obj_loc in objs]
        final_state_objs = [obj_loc[0] for objs in final_state.values() for obj_loc in objs]
        ablated_objects = set(initial_state_objs).symmetric_difference(set(final_state_objs))
        objects_to_put_away = list(ablated_objects)
        scenario = {
            "objects_to_put_away": objects_to_put_away,
            "initial_state": initial_state
        }
        scenarios.append(scenario)
        viz_data.append((objects_to_put_away, best_action_sequence, best_obj_names))
        i += 1
    return scenarios, viz_data

def specific_location_generator(num_tests, seed):
    """Returns a dictionary of test cases for the specific location preference.
    
    The test case format is as follows:
    {
        "0": {
            "type": "specific_location",
            "gt_preference": ...,
            "demonstrations": [
                {
                    "objects_to_put_away": [...],
                    "initial_state": {
                        "location": [
                            ["name", [w, h]],
                            ...
                        ],
                        ...
                    },
                    "final_state": {...}
                },
                ...
            ],
            "scenarios": [
                {
                    "objects_to_put_away": [...],
                    "initial_state": {...}
                },
                ...
            ]
        },
        "1": {...},
        ...
    }

    These test cases are particularly for preferences where the user specifies the exact 
    location of each object with respect to the fridge. There are two demonstrations that
    have enough coverage to convey the user's preference and two unique scenarios for each
    preference.

    Parameters:
        num_tests (int): The number of test cases to generate.
        seed (int): The seed for the random number generator.
    
    Returns:
        tests (dict): The test cases in the format specified above.
        viz_data (dict): The visualization data for the demonstrations and scenarios.
    """
    random.seed(seed)
    env = sim2d_utils.make_sim2d_env(render_mode="rgb_array") # TODO(chalo2000): Allow setting environment to initial state
    gt_preference_dict, gt_preference_str = generate_ground_truth_preference()
    tests = {}
    viz_data = {}
    for i in range(num_tests):
        demonstrations, demo_viz_data = generate_demonstrations(env, gt_preference_dict, num_demonstrations=2, demo_objects=7, obj_to_place=4)
        # input(f"Generated demonstrations for Test {i}. Press Enter to continue.")
        scenarios, scenario_viz_data = generate_scenarios(env, gt_preference_dict, num_scenarios=2, scenario_objects=10, obj_to_place=6)
        # input(f"Generated scenarios for Test {i}. Press Enter to continue.")
        test = {
            "type": "specific_location",
            "gt_preference": gt_preference_str,
            "demonstrations": demonstrations,
            "scenarios": scenarios
        }
        tests[str(i)] = test
        viz_data[str(i)] = {
            "demonstrations": demo_viz_data,
            "scenarios": scenario_viz_data
        }
    test_formatter(tests)
    return tests, viz_data

def test_formatter(tests):
    """Formats the test cases to be JSON serializable.
    
    Parameters:
        tests (dict): The test cases to format.
    
    Side Effects:
        Modifies the input dictionary in place.
    """
    formatted_tests = {}
    locations = ["left side of top shelf", "right side of top shelf", "left side of middle shelf", "right side of middle shelf", "left side of bottom shelf", "right side of bottom shelf"]
    for key, test in tests.items():
        for demonstration in test["demonstrations"]:
            ordered_initial_state = [(location, demonstration['initial_state'].get(location, ())) for location in locations]
            demonstration["initial_state"] = OrderedDict(ordered_initial_state)
            ordered_final_state = [(location, demonstration['final_state'].get(location, ())) for location in locations]
            demonstration["final_state"] = OrderedDict(ordered_final_state)
        for scenario in test["scenarios"]:
            ordered_initial_state = [(location, scenario['initial_state'].get(location, ())) for location in locations]
            scenario["initial_state"] = OrderedDict(ordered_initial_state)
    return formatted_tests

def save_replay_for_viz(env, viz, gif_path):
    """Saves a replay for the visualization data.
    
    Rearranges action_sequence such that the indices of obj_names corresponding to objs_to_put_away are last

    Parameters:
        env (gym.Env): The environment.
        viz (list): The visualization data.
        gif_path (str): The path to save the GIF.
    
    Side Effects:
        Saves a GIF to the specified path.
    """
    objs_to_put_away, action_sequence, obj_names = viz
    actions_to_put_away = []
    actions_for_in_fridge = []
    for obj_name in objs_to_put_away:
        action = action_sequence[obj_names.index(obj_name)]
        actions_to_put_away.append(action)
    for action in action_sequence:
        if action not in actions_to_put_away:
            actions_for_in_fridge.append(action)
    action_sequence = actions_for_in_fridge + actions_to_put_away
    sim2d_utils.save_replay(env, action_sequence, gif_path, action_to_start=len(actions_for_in_fridge))

def visualize_data(viz_data):
    """Visualizes the demonstrations and scenarios in the visualization data.
    
    Parameters:
        viz_data (dict): The visualization data to visualize.
    """
    env = sim2d_utils.make_sim2d_env(render_mode="rgb_array")
    for i, viz in viz_data.items():
        demonstrations = viz["demonstrations"]
        scenarios = viz["scenarios"]
        for j, demonstration in enumerate(demonstrations):
            save_replay_for_viz(env, demonstration, f"generated_tests/specific_location/demonstrations/test_{i}_demonstration_{j}.gif")
        for j, scenario in enumerate(scenarios):
            save_replay_for_viz(env, scenario, f"generated_tests/specific_location/scenarios/test_{i}_scenario_{j}.gif")

if __name__ == "__main__":
    tests, viz_data = specific_location_generator(1, 0)

    # Write tests to file
    with open("generated_tests/specific_location/tests.json", "w") as file:
        json.dump(tests, file, indent=4)
    
    # Visualize demonstrations and scenarios
    visualize_data(viz_data)