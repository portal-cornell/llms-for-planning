"""The V0 single heuristic planner uses an LLM to generate a single plan to reach a goal in a given
environment. Then, starting from an initial state, the plan is used to propose an action(s) to take
in order to reach the goal. Using a model of the environment, the next states are computed and the
plan is used to select the next state to propose actions from. This process is repeated until the
goal is reached.

The V0 single heuristic planner is the simplest planner that we'll use to test the effectiveness
of an LLM as an action proposer and state selector.
"""
import argparse
from copy import deepcopy
import networkx as nx
import random
import imageio
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import pddlgym_utils

def save_env_render(env, state):
    """Saves the environment render to a file whose name is returned.

    Parameters:
        env (gym.Env)
            The environment to render.
        state (object)
            The state of the environment to render.
    
    Returns:
        file_name (str)
            The name of the file that the render was saved to.
    """
    img = env.render()
    plt.close()
    state_hash = hash(state)
    imageio.imsave(f'./images/{state_hash}.png', img)
    return f'./images/{state_hash}.png'

def generate_plan():
    # TODO: Use LLM
    return None

def propose_actions(env, state, plan):
    """Proposes an action(s) to take in order to reach the goal.

    Parameters:
        env (gym.Env)
            The environment to propose actions in.
        state (object)
            The current state of the environment.
        plan (object)
            The plan to use to propose actions.
    
    Returns:
        actions (list)
            The action(s) to take in order to reach the goal.
    """
    # TODO: Use LLM
    valid_actions = pddlgym_utils.get_valid_actions(env, state)
    # for i, action in enumerate(valid_actions):
    #         print(f"{i}: {action}")
    # input_action = input("Enter the index of the action to take: ")
    # action = valid_actions[int(input_action)]
    return [random.choice(valid_actions)]

def compute_next_states(graph, env, current_state, actions):
    """Computes the next states to add as nodes in the graph with directed action edges from the current state.

    Parameters:
        graph (nx.DiGraph)
            The graph to add the next states to.
        env (gym.Env)
            The environment to simulate the actions in.
        current_state (object)
            The current state of the environment.
        actions (list)
            The actions to simulate in the environment.
    
    Side Effects:
        Modifies the graph by adding the next states as nodes and the actions as edges.
    """
    for action in actions:
        env_copy = deepcopy(env)
        next_state, _, _, _, _ = env_copy.step(action)
        img_path = save_env_render(env_copy, next_state)
        graph.add_node(next_state, label="", image=img_path, env=env_copy)
        graph.add_edge(current_state, next_state, action=action)

def select_state(env, states, plan):
    # TODO: Use LLM
    return random.choice(list(states))

def plan(env, initial_state, goal, max_steps=20):
    """Follows the V0 single heuristic planning algorithm to output a sequence of actions to the goal."""

    # Generate plan
    plan = generate_plan()

    # Follow plan to reach goal
    graph = nx.DiGraph()
    img_path = save_env_render(env, initial_state)
    graph.add_node(initial_state, label="", image=img_path, env=deepcopy(env))
    selected_state = initial_state
    steps = 0
    pbar = tqdm(total=max_steps)
    while not pddlgym_utils.did_reach_goal(selected_state, goal) and steps < max_steps:
        curr_env = graph.nodes[selected_state]["env"]
        # Propose actions
        actions = propose_actions(curr_env, selected_state, plan)
        # Compute the next states to add as nodes in the graph with directed action edges from the current state
        compute_next_states(graph, curr_env, selected_state, actions)
        # Select next state
        selected_state = select_state(curr_env, graph.nodes, plan)
        steps += 1
        pbar.update(1)
    
    # Get the shortest action sequence to the goal
    shortest_path = nx.shortest_path(graph, initial_state, selected_state)
    action_sequence = []
    for i in range(len(shortest_path) - 1):
        current_state = shortest_path[i]
        next_state = shortest_path[i + 1]
        action = graph[current_state][next_state]["action"]
        action_sequence.append(action)
    
    return action_sequence, graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", required=True, help="The name of the environment.")
    parser.add_argument("--graph_file", required=True, help="The name of the file to save the graph to.")
    args = parser.parse_args()
    env_name = f"PDDLEnv{args.env_name.capitalize()}-v0"
    env = pddlgym_utils.make_pddlgym_env(env_name)
    initial_state, _ = env.reset()
    goal = initial_state.goal
    action_sequence, graph = plan(env, initial_state, goal)

    # Draw graph
    pygraphviz_graph = nx.nx_agraph.to_agraph(graph)
    pygraphviz_graph.layout('dot')
    pygraphviz_graph.draw(args.graph_file)