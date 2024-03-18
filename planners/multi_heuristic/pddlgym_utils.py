from copy import deepcopy
import imageio
import matplotlib.pyplot as plt
import random
import tempfile

import pddlgym

def make_pddlgym_env(env_name):
    """Returns the PDDLGym environment with the given name.
    
    Parameters:
        env_name (str)
            The name of the PDDLGym environment to make.
    
    Returns:
        env (gym.Env)
            The PDDLGym environment with the given name.
    """
    return pddlgym.make(env_name)

def get_valid_actions(env, obs):
    """Returns the valid actions for the given observation.
    
    Parameters:
        env (gym.Env)
            The environment.
        obs (object)
            The observation.
    
    Returns:
        valid_actions (list)
            The valid actions for the given observation.
    """
    all_actions = env.action_space.all_ground_literals(obs)
    all_actions = sorted(all_actions)
    valid_actions = []
    for action in all_actions:
        env_copy = deepcopy(env)
        next_obs, _, _, _, _ = env_copy.step(action)
        if obs != next_obs:
            valid_actions.append(action)
    return valid_actions

def did_reach_goal(state, goal):
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

def get_image_path(env):
    """Saves the environment render to a file whose name is returned.

    Parameters:
        env (gym.Env)
            The environment to render.
    
    Returns:
        file_name (str)
            The name of the file that the render was saved to.
    """
    img = env.render()
    plt.close()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        imageio.imsave(temp_file.name, img)
        return temp_file.name

def render_pddlgym(env, step_time, render=False, close=False):
    """Renders the environment and returns the image.
    
    Parameters:
        close (bool)
            Whether to close the previous PDDLGym window before rendering.
        step_time (float)
            The time to pause between steps.
        render (bool)
            Whether to render the environment through Matplotlib.
        env (gym.Env)
            The environment.
    
    Returns:
        img (np.ndarray)
            The image of the environment.
    """
    if close: plt.close()
    img = env.render()
    if render:
        plt.gcf().set_size_inches(9, 9)
        plt.pause(step_time)
    return img

def get_action(env, obs, mode):
    """Returns the action to take in the environment.
    
    Parameters:
        env (gym.Env)
            The environment.
        obs (object)
            The observation.
        mode (str)
            The mode to use to select the action. Options include ["random", "interactive"].
    
    Returns:
        action (object)
            The action to take in the environment.
    """
    valid_actions = get_valid_actions(env, obs)
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
    env = make_pddlgym_env(env_name)
    obs, _ = env.reset()
    # Render
    if render:
        plt.ion()
        plt.show()
    img = render_pddlgym(env, step_time, render=render)
    imgs = [img]
    # Simulate steps
    i = 0
    done = False
    while not done and i < max_steps:
        action = get_action(env, obs, mode)
        obs, reward, terminated, truncated, info = env.step(action)
        print(did_reach_goal(obs, obs.goal))
        done = terminated or truncated
        i += 1
        # Render
        img = render_pddlgym(env, step_time, render=render, close=True)
        imgs.append(img)
    if save_gif:
        imageio.mimsave(f"{env_name}.gif", imgs, duration=max_steps)