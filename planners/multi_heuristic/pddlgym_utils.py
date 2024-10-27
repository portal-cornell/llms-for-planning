"""
This module contains utility functions for using PDDLGym environments.
"""
import os
from copy import deepcopy
import imageio
import matplotlib.pyplot as plt
import random
import tempfile
import shutil

import pddlgym
from pddlgym_planners.fd import FD
from pddlgym_planners.ff import FF

from robotouille.env import LanguageSpace
from robotouille.robotouille_env import create_robotouille_env

# TODO(chalo2000): Move to separate location (along with the models)
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
    pddlgym_problem_dir_path = add_problem_files(env_name, problems_dir_path)
    env = pddlgym.make(f"PDDLEnv{env_name.capitalize()}-v0")
    os.remove(pddlgym_domain_file_path)
    shutil.rmtree(pddlgym_problem_dir_path)
    return env

# TODO(chalo2000): Move Model and PDDLGymModel to a separate location
class Model:
    """An interface for a model that takes in an environment and returns information for a planner."""

    def __init__(self, env, **kwargs):
        """Initializes the model.

        Parameters:
            env (gym.Env)
                The environment to use for the model.
            kwargs (dict)
                The keyword arguments for the model.
        """
        self.env = env
        self.kwargs = kwargs
    
    def get_valid_actions(self, state):
        """Returns the valid actions for the given state.

        Parameters:
            state (object)
                The state to get valid actions for

        Returns:
            valid_actions (list)
                The valid actions for the given state
        """
        raise NotImplementedError

    def did_reach_goal(self, state, goal):
        """Returns whether the given state satisfies the given goal.
        
        Parameters:
            state (object)
                The state to check.
            goal (object)
                The goal to satisfy.
        
        Returns:
            reached_goal (bool)
                Whether the given state satisfies the given goal.
        """
        raise NotImplementedError
    
    def get_image_path(self):
        """Returns the path to an image of the environment's current state
        
        Returns:
            image_path (str)
                The path to an image of the environment's current state
        """
        raise NotImplementedError
    
    def state_to_str(self, state):
        """Returns a string representation of the state.
        
        Parameters:
            state (object)
                The state to convert to a string.
        
        Returns:
            state_str (str)
                The string representation of the state.
        """
        raise NotImplementedError
    
class PDDLGymModel(Model):
    """A model for PDDLGym environments."""
    
    def __init__(self, env, **kwargs):
        """Initializes the PDDLGym model.
        
        Parameters:
            env (gym.Env)
                The environment to use for the model.
            kwargs (dict)
                The keyword arguments for the model.
        """
        super().__init__(env, **kwargs)
    
    def get_valid_actions(self, state):
        """Returns the valid actions for the given state.
        
        Parameters:
            state (object)
                The state to get valid actions for
        
        Returns:
            valid_actions (list)
                The valid actions for the given state
        """
        all_actions = self.env.action_space.all_ground_literals(state)
        all_actions = sorted(all_actions)
        valid_actions = []
        for action in all_actions:
            env_copy = deepcopy(self.env)
            next_state, _, _, _, _ = env_copy.step(action)
            if state != next_state:
                valid_actions.append(action)
        return valid_actions
    
    def did_reach_goal(self, state, goal):
        """Returns whether the given state satisfies the given goal.
        
        Parameters:
            state (object)
                The state to check.
            goal (object)
                The goal to satisfy.
        
        Returns:
            reached_goal (bool)
                Whether the given state satisfies the given goal.
        """
        return pddlgym.inference.check_goal(state, goal)
    
    def get_image_path(self):
        """Returns the path to an image of the environment's current state
        
        Returns:
            image_path (str)
                The path to an image of the environment's current state
        """
        img = self.env.render()
        plt.close()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            imageio.imsave(temp_file.name, img)
            return temp_file.name
    
    def state_to_str(self, state):
        """Returns a string representation of the state.
        
        Parameters:
            state (object)
                The state to convert to a string.
        
        Returns:
            state_str (str)
                The string representation of the state.
        """
        literals = [str(literal) for literal in state.literals]
        objects = [str(obj) for obj in state.objects]
        str_state = f"""Predicates: {', '.join(literals)}
        Objects: {', '.join(objects)}"""
        return str_state

    def goal_to_str(self, state, goal):
        """Returns a string representation of the goal.
        
        Parameters:
            state (object)
                The state to check.
            goal (object)
                The goal to convert to a string.
        
        Returns:
            goal_str (str)
                The string representation of the goal.
        """
        # Returns a dictionary whose keys are the goal predicates and values are whether or not they are negated
        str_literals = [str(literal) for literal in goal.literals]
        return f"Goal: {', '.join(str_literals)}"
        # import pdb; pdb.set_trace()
        # goal_dict = goal.as_dict()
        # goal_str = f"Goal: {goal_dict}"
        # return goal_str
        # goal_list = []
        # for literal in goal.literals:
        #     goal_satisfied = (str(literal), literal in state.literals)
        #     goal_list.append(goal_satisfied)
        # # random.shuffle(goal_list) # Order doesn't matter - shuffling to avoid bias
        # goal_str = "\n".join([f"- {literal}: {satisfied}" for literal, satisfied in goal_list])
        return goal_str

class RobotouilleModel(Model):
    """A model for Robotouille environments."""
    
    def __init__(self, env, **kwargs):
        """Initializes the Robotouille model.
        
        Parameters:
            env (gym.Env)
                The environment to use for the model.
            kwargs (dict)
                The keyword arguments for the model.
        """
        super().__init__(env, **kwargs)
    
    def get_valid_actions(self, state):
        """Returns the valid actions for the given state.
        
        Parameters:
            state (object)
                The state to get valid actions for
        
        Returns:
            valid_actions (list)
                The valid actions for the given state
            str_valid_actions (list)
                The string representation of the valid actions for the given state
        """
        return state.get_valid_actions_and_str()
    
    def did_reach_goal(self, state, goal):
        """Returns whether the given state satisfies the given goal.
        
        Parameters:
            state (object)
                The state to check.
            goal (object)
                The goal to satisfy.
        
        Returns:
            reached_goal (bool)
                Whether the given state satisfies the given goal.
        """
        return self.env.current_state.is_goal_reached()
    
    def get_image_path(self):
        """Returns the path to an image of the environment's current state
        
        Returns:
            image_path (str)
                The path to an image of the environment's current state
        """
        img = self.env.render("rgb_array")
        plt.close()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            imageio.imsave(temp_file.name, img)
            return temp_file.name
    
    def state_to_str(self, state):
        """Returns a string representation of the state.
        
        Parameters:
            state (object)
                The state to convert to a string.
        
        Returns:
            state_str (str)
                The string representation of the state.
        """
        return LanguageSpace.state_to_language_description(state)

    def goal_to_str(self, state, goal):
        """Returns a string representation of the goal.
        
        Parameters:
            state (object)
                The state to check.
            goal (object)
                The goal to convert to a string.
        
        Returns:
            goal_str (str)
                The string representation of the goal.
        """
        assert False, "Goal included in state, so do not call this function"

def make_pddlgym_model(env_name=None, domain_file=None, instance_dir=None, render_fn_name=None):
    """Returns the model for the PDDLGym environment with the given name.
    
    Parameters:
        env_name (Optional[str])
            The name of the PDDLGym environment to make.
        domain_file (Optional[str])
            The path to the domain file.
        instance_dir (Optional[str])
            The directory containing the problem PDDL files.
        render_fn_name (Optional[str])
            The name of the function to render the environment.
    
    Returns:
        model (PDDLGymModel)
            The model for the PDDLGym environment
    """
    if domain_file and instance_dir:
        env = create_pddl_env(domain_file, instance_dir, render_fn_name)
    elif env_name:
        env = pddlgym.make(env_name)
    else:
        raise ValueError("Either env_name or (domain_dir, instance_dir, render_fn_name) must be provided.")
    model = PDDLGymModel(env)
    return model

def make_robotouille_model(env_name, seed=None, noisy_randomization=False):
    """Returns the model for the Robotouille environment with the given name.
    
    Parameters:
        env_name (Optional[str])
            The name of the Robotouille environment to make.
        seed (Optional[int])
            The seed to use for randomization.
        noisy_randomization (Optional[bool])
            Whether to use noisy randomization.
    
    Returns:
        model (RobotouilleModel)
            The model for the Robotouille environment
    """
    env = create_robotouille_env(env_name, seed=seed, noisy_randomization=noisy_randomization)
    model = RobotouilleModel(env)
    return model

def render_pddlgym(model, step_time, render=False, close=False):
    """Renders the environment and returns the image.
    
    Parameters:
        close (bool)
            Whether to close the previous PDDLGym window before rendering.
        step_time (float)
            The time to pause between steps.
        render (bool)
            Whether to render the environment through Matplotlib.
        model (PDDLGymModel)
            The model containing the environment.
    
    Returns:
        img (np.ndarray)
            The image of the environment.
    """
    if close: plt.close()
    img = model.env.render()
    if render:
        plt.gcf().set_size_inches(9, 9)
        plt.pause(step_time)
    return img

def get_action(model, obs, mode):
    """Returns the action to take in the environment.
    
    Parameters:
        model (PDDLGymModel)
            The model to get an action from.
        obs (object)
            The observation.
        mode (str)
            The mode to use to select the action. Options include ["random", "interactive"].
    
    Returns:
        action (object)
            The action to take in the environment.
    """
    valid_actions = model.get_valid_actions(obs)
    if mode == "random":
        action = random.choice(valid_actions)
    elif mode == "interactive":
        for i, action in enumerate(valid_actions):
            print(f"{i}: {action}")
        input_action = ""
        while not input_action.isdigit() or int(input_action) >= len(valid_actions):
            input_action = input("Enter action idx: ")
        action = valid_actions[int(input_action)]
    return action

def play_env(env_name, max_steps=100, step_time=0.5, fps=4, mode="random", render=False, gif_file=None):
    """Play the environment with the given mode.
    
    Parameters:
        env_name (str)
            The name of the PDDLGym environment to play.
        max_steps (int)
            The maximum number of steps to simulate.
        step_time (float)
            The time to pause between steps.
        fps (int)
            The frames per second to save the gif as.
        mode (str)
            The mode to use to select the action. Options include ["random", "interactive"].
        render (bool)
            Whether to render the environment.
        gif_file (Optional[str])
            The name of the file to save the gif to, if any.
    """
    # Initialize environment
    model = make_pddlgym_model(env_name)
    # Fix problem with PDDLGym environments
    model.env.fix_problem_index(0)
    obs, _ = model.env.reset()
    # Render
    if render:
        plt.ion()
        plt.show()
    img = render_pddlgym(model, step_time, render=render)
    imgs = [img]
    # Simulate steps
    i = 0
    done = False
    while not done and i < max_steps:
        action = get_action(model, obs, mode)
        obs, reward, terminated, truncated, info = model.env.step(action)
        print(model.did_reach_goal(obs, obs.goal))
        done = terminated or truncated
        i += 1
        # Render
        img = render_pddlgym(model, step_time, render=render, close=True)
        imgs.append(img)
    if gif_file:
        os.makedirs(os.path.dirname(gif_file), exist_ok=True)
        imageio.mimsave(gif_file, imgs, fps=4)

def get_optimal_plan(domain, initial_state, alias="a*-lmcut"):
    """Returns the optimal plan for the environment using the Fast Downward planner.
    
    Parameters:
        domain (PDDLProblemParser)
            The domain for an environment
        initial_state (State)
            The initial state of the environment.
        alias (str)
            The alias to use for the planner.
    
    Returns:
        plan (List[pddlgym.structs.Literal])
            The optimal plan for the environment.
        statistics (dict)
            The statistics for the planner.
    """
    planner = FD(alias_flag=f"--alias {alias}")
    plan = planner(domain, initial_state)
    return plan, planner._statistics