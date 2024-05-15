"""
This script takes in a domain file and directory of problem files to copy and save into directories 
named by the number of steps in the optimal plan.

To run this script on your data, run the following command in the terminal:
    python horizon_dir_script.py \
        --domain_file <DOMAIN_FILE> \
        --problem_dir <PROBLEM_DIR> \
        --save_dir <SAVE_DIR>
"""
import argparse
import os
import re
import shutil
from tqdm import tqdm, trange

import planners.multi_heuristic.pddlgym_utils as pddlgym_utils

def get_problem_name(problem_file):
    """Returns the problem name from the problem file.
    
    Parameters:
        problem_file (str)
            The path to the problem file.
    
    Returns:
        problem_name (Optional[str])
            The name of the problem, or None if it could not be found.
    
    Side Effects:
        - Reads the contents of the problem file
    """
    regex = r'\(problem (.+)\)'
    with open(problem_file, "r") as f:
        file_contents = f.read()
    problem_name = re.search(regex, file_contents)
    if problem_name is None: return None
    return problem_name.group(1)

def get_num_files_for_env(env_name, new_dir):
    """Returns the number of files for the environment in the directory.
    
    Parameters:
        env_name (str)
            The name of the environment.
        new_dir (str)
            The directory to search for files.
    
    Returns:
        num_files (int)
            The number of files in the directory with the environment name.
    """
    regex = r'(.+)_instance'
    num_files = 0
    for f in os.listdir(new_dir):
        match = re.search(regex, f)
        if match is not None:
            num_files += match.group(1) == env_name
    return num_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_file", type=str, required=True, help="The domain PDDL to use.")
    parser.add_argument("--problem_dir", type=str, required=True, help="The directory containing the problem PDDL files.")
    parser.add_argument("--save_dir", type=str, required=True, help="The directory to save the new problem PDDL files.")
    args = parser.parse_args()

    # Create pddlgym environment
    model = pddlgym_utils.make_pddlgym_model(domain_file=args.domain_file, instance_dir=args.problem_dir)

    # Setup path length dict
    path_lengths = {}
    problems = model.env.problems
    for i in trange(len(problems)):
        model.env.fix_problem_index(i)
        initial_state, _ = model.env.reset()
        optimal_plan, _ = pddlgym_utils.get_optimal_plan(model.env.domain, initial_state)
        optimal_plan_length = len(optimal_plan)
        problem_fname = os.path.basename(problems[i].problem_fname)
        path_lengths[optimal_plan_length] = path_lengths.get(optimal_plan_length, []) + [problem_fname]
    
    for _, (k, v) in enumerate(tqdm(path_lengths.items())):
        new_dir = os.path.join(args.save_dir, str(k))
        os.makedirs(new_dir, exist_ok=True)
        
        problem_numfiles = {}
        for _, filename in enumerate(tqdm(v)):
            problem_file = os.path.join(args.problem_dir, filename)
            env_name = get_problem_name(problem_file)
            if env_name is None:
                print(f"Could not find problem name in {problem_file}")
                continue
            num_files = problem_numfiles.get(env_name, get_num_files_for_env(env_name, new_dir))
            new_path = os.path.join(new_dir, f"{env_name}_instance-{num_files+1}.pddl")
            problem_numfiles[env_name] = num_files + 1
            shutil.copy2(problem_file, new_path)