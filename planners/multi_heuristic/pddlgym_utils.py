from copy import deepcopy
import imageio
import matplotlib.pyplot as plt
import random
import tempfile

import pddlgym

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
    
def make_pddlgym_model(env_name):
    """Returns the model for the PDDLGym environment with the given name.
    
    Parameters:
        env_name (str)
            The name of the PDDLGym environment to make.
    
    Returns:
        model (PDDLGymModel)
            The model for the PDDLGym environment
    """
    env = pddlgym.make(env_name)
    model = PDDLGymModel(env)
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

def play_env(env_name, max_steps=100, step_time=0.5, mode="random", render=False, save_gif=False):
    """Play the environment with the given mode.
    
    Parameters:
        env_name (str)
            The name of the PDDLGym environment to play.
        max_steps (int)
            The maximum number of steps to simulate.
        step_time (float)
            The time to pause between steps.
        mode (str)
            The mode to use to select the action. Options include ["random", "interactive"].
        render (bool)
            Whether to render the environment.
        save_gif (bool)
            Whether to save the rendered environment as a GIF.
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
    if save_gif:
        imageio.mimsave(f"{env_name}.gif", imgs, duration=max_steps)