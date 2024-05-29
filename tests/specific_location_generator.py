import random

import sys
sys.path.append('../planners/geometric_feasibility/') # Hack to import plan from v0_no_llm_scoring
import v0_no_llm_scoring # Must uncomment `import sim2d_utils` in v0_no_llm_scoring.py to work
import sim2d_utils

CATEGORY_DICT = {
	"fruits": ["apple", "banana", "cherries", "orange", "pear", "watermelon", "papaya", "avocado", "mango", "pomegranate", "honeydew"],
	"dairy products": ["almond_milk", "oat_milk", "whole_milk", "cheese_block", "yogurt", "butter", "egg", "sliced_cheese", "ice_cream", "string_cheese"],
	"juice and soft drinks": ["fanta", "pepsi", "coca_cola", "ginger_ale", "gatorade", "strawberry_milk", "banana_milk", "orange_juice", "apple_juice", "cranberry_juice"],
	"condiments": ["chocolate_sauce", "ketchup", "mustard", "salad_dressing", "soy_sauce", "hot sauce", "nutella", "maple_syrup", "whipped_cream", "peanut_butter"],
	"vegetables": ["lettuce", "onion", "potato", "tomato", "eggplant", "pumpkin", "bell_pepper", "carrot", "radish", "corn"]
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
    gt_preference_str = "I want "
    for category in CATEGORY_DICT.keys():
        direction = random.choice(["left", "right"])
        shelf = random.choice(["top", "middle", "bottom"])
        location = f"{direction} side of {shelf} shelf"
        gt_preference_dict[category] = location
        gt_preference_str += f"{category} on the {location}, "
    gt_preference_str = gt_preference_str[:-2] + "."
    return gt_preference_dict, gt_preference_str

def generate_demonstrations(env, gt_preference_dict, num_demonstrations=2, demo_objects=7):
    """Returns a list of demonstrations for the specific location preference.
    
    Parameters:
        env (gym.Env): The environment to generate the demonstrations in.
        gt_preference_dict (dict): The ground truth preference dictionary.
        num_demonstrations (int): The number of demonstrations to generate.
        demo_objects (int): The number of objects to put away in each demonstration.
    
    Returns:
        demonstrations (list): A list of demonstrations.
    """
    demonstrations = []
    ordered_categories = list(CATEGORY_DICT.keys())
    i = 0
    while i < num_demonstrations:
        # Generate 'demo_objects' objects to put away that encompass all categories
        num_categories = len(ordered_categories)
        extra_objects = demo_objects - num_categories
        demo_categories = ordered_categories + random.choices(ordered_categories, k=extra_objects)
        objects_to_put_away = [random.choice(CATEGORY_DICT[category]) for category in demo_categories]
        
        # Generate the final state of the fridge
        text_plan = ""
        for object, category in zip(objects_to_put_away, demo_categories):
            # Construct text plan
            location = gt_preference_dict[category]
            text_skill = f'pickandplace("{object}", "{location}")'
            text_plan += text_skill + "\n"
        text_plan = text_plan[:-1] # Remove last newline character
        # Execute text plan
        env.reset()
        best_action_sequence, best_obj_names = v0_no_llm_scoring.plan(env, 1, 10, 10, text_plans=[text_plan], perception_values=None)
        import pdb; pdb.set_trace()

        # Ablate the final state to generate the initial state
        i += 1
    return demonstrations

def generate_scenarios(gt_preference_dict):
    pass
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
    """
    random.seed(seed)
    env = sim2d_utils.make_sim2d_env(render_mode="rgb_array") # TODO(chalo2000): Allow setting environment to initial state
    gt_preference_dict, gt_preference_str = generate_ground_truth_preference()
    tests = {}
    for i in range(num_tests):
        demonstrations = generate_demonstrations(env, gt_preference_dict)
        scenarios = generate_scenarios(gt_preference_dict)
        test = {
            "type": "specific_location",
            "gt_preference": gt_preference_str,
            "demonstrations": demonstrations,
            "scenarios": scenarios
        }
        tests[str(i)] = test
    return tests

if __name__ == "__main__":
    tests = specific_location_generator(1, 0)
    print(tests)