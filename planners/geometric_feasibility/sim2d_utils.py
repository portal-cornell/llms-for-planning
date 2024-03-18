from copy import deepcopy
import imageio
import gym
import random

import grocery_perception.planning_sims

def make_sim2d_env(render_mode="human"):
    """Returns the Sim2D environment.

    Parameters:
        render_mode (str)
            The mode to use to render the environment. Options include ["human", "rgb_array"].
    
    Returns:
        env (gym.Env)
            The Sim2D environment.
    """
    return gym.make("PlanSim2D-v0", render_mode=render_mode)

def check_collision(env, action):
    """Returns whether the action results in a collision in the environment.
    
    Parameters:
        env (gym.Env)
            The environment.
        action (object)
            The action to take in the environment.
    
    Returns:
        collision (bool)
            Whether the action results in a collision in the environment.
    """
    env_copy = deepcopy(env)
    _, _, _, _, info = env_copy.step(action)
    return info.get("collision") is not None

def get_shelf_heights():
    """Returns the heights of the shelves in the environment.
    
    Returns:
        shelf_heights (dict)
            A dictionary mapping the shelf number to its height.
    """
    return {
        "bottom": 0.0,
        "middle": 0.33,
        "top": 0.66
    }

def get_action(env, mode):
    """Returns the action to take in the environment.
    
    Parameters:
        env (gym.Env)
            The environment.
        mode (str)
            The mode to use to select the action. Options include ["random", "interactive"].
    
    Returns:
        action (object)
            The action to take in the environment.
    """
    if mode == "random":
        collision = True
        while collision:
            x = random.uniform(0, 1)
            y = random.choice(list(get_shelf_heights().values()))
            w = random.uniform(0.1, 0.3)
            h = random.uniform(0.1, 0.3)
            action = (x, y, w, h)
            collision = check_collision(env, action)
    elif mode == "interactive":
        collision = True
        input_action = ()
        while collision or len(input_action) != 4 or not all(isinstance(x, float) for x in input_action):
            try:
                x_action = float(input("Enter x: "))
                y_action = float(input("Enter y: "))
                w_action = float(input("Enter width: "))
                h_action = float(input("Enter height: "))
                r_action = float(input("Enter red color: "))
                g_action = float(input("Enter green color: "))
                b_action = float(input("Enter blue color: "))
                input_action = (x_action, y_action, w_action, h_action, (r_action, g_action, b_action))
            except ValueError:
                print("Invalid action. Please try again.")
            collision = check_collision(env, input_action)
        action = input_action
    return action

def play_env(max_steps=100, mode="random", render_mode="human", gif_path=None):
    """Play the Sim2D environment with the given mode.
    
    Parameters:
        max_steps (int)
            The maximum number of steps to simulate.
        mode (str)
            The mode to use to select the action. Options include ["random", "interactive"].
        render_mode (str)
            The mode to use to render the environment. Options include ["human", "rgb_array"].
        gif_path (str)
            The path to save the GIF to; doesn't save if None. Requires render_mode="rgb_array".
    """
    assert gif_path is None or render_mode == "rgb_array", "GIF requires rgb_array rendering"
    # Initialize environment
    env = make_sim2d_env(render_mode=render_mode)
    obs, _ = env.reset()
    # Render
    img = env.render()
    imgs = [img]
    # Simulate steps
    i = 0
    done = False
    while not done and i < max_steps:
        action = get_action(env, mode)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        i += 1
        # Render
        img = env.render()
        imgs.append(img)
        print(info)
    if gif_path is not None:
        imageio.mimsave(gif_path, imgs, duration=max_steps)

def save_replay(env, actions, gif_path):
    """Saves a replay of the environment with the given actions to the given GIF path.
    
    Parameters:
        env (gym.Env)
            The environment.
        actions (list)
            The list of actions to take in the environment.
        gif_path (str)
            The path to save the GIF to.
    """
    # Initialize environment
    env = deepcopy(env)
    env.reset()
    # Render
    img = env.render()
    imgs = [img]
    # Simulate steps
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        # Render
        img = env.render()
        imgs.append(img)
    imageio.mimsave(gif_path, imgs, duration=len(actions))