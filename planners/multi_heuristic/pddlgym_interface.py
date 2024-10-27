import pddlgym
import os
import shutil

CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PDDL_DIR_PATH = os.path.join(CURRENT_DIR_PATH, "pddlgym", "pddlgym", "pddl")

def add_domain_file(domain_file_path):
    """
    Adds a domain file to the PDDL directory.

    Parameters:
        domain_file_path (str):
            Path to the domain file
    
    Returns:
        pddlgym_domain_file_path (str): Path to the domain file in the PDDL directory
    """
    shutil.copy2(domain_file_path, PDDL_DIR_PATH)
    domain_file_name = os.path.basename(domain_file_path)
    pddlgym_domain_file_path = os.path.join(PDDL_DIR_PATH, domain_file_name)
    return pddlgym_domain_file_path

def add_problem_files(env_name, problem_dir_path):
    """
    Add problem files to the PDDL directory.

    Parameters:
        env_name (str):
            Name of the environment (domain file name without the extension)
        problem_dir_path (str):
            Path to the directory containing the problem files
    
    Returns:
        pddlgym_problem_dir_path (str):
            Path to the problem files in the PDDL directory
    """
    pddl_problem_dir_path = os.path.join(PDDL_DIR_PATH, env_name)
    if os.path.exists(pddl_problem_dir_path):
        shutil.rmtree(pddl_problem_dir_path)
    shutil.copytree(problem_dir_path, pddl_problem_dir_path)
    return pddl_problem_dir_path

def create_pddl_env(domain_file_path, problems_dir_path, render_fn_name):
    """
    Creates and returns a PDDLGym environment.

    PDDLGym requires a domain file and problem files to be in the PDDL directory. We
    temporarily copy the necessary files to that directory and delete them after the
    environment has been created.

    Parameters:
        domain_file_path (str):
            Path to the domain file
        problems_dir_path (str):
            Path to the directory containing the problem files
        render_fn_name (str):
            The name of the function to render the environment
        problem_filename (str):
            Name of the problem file to generate the environment for
    
    Returns:
        env (pddlgym.PDDLEnv): PDDLGym environment
    """
    env_name = os.path.basename(domain_file_path).split(".")[0]
    # Fixed PDDLGym settings
    is_test_env = False
    render_fn = None
    if render_fn_name == "blocksworld":
        render_fn = pddlgym.rendering.blocks_render
    kwargs = {
        'render': render_fn,
        'operators_as_actions': True, 
        'dynamic_action_space': True,
        "raise_error_on_invalid_action": False
        }
    pddlgym.register_pddl_env(env_name, is_test_env, kwargs)

    pddlgym_domain_file_path = add_domain_file(domain_file_path)
    pddlgym_problem_dir_path = add_problem_files(problems_dir_path)
    env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0")
    os.remove(pddlgym_domain_file_path)
    shutil.rmtree(pddlgym_problem_dir_path)
    return env