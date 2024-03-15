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
from tqdm import tqdm

import pddlgym_utils

def generate_plan():
    # TODO: Use LLM
    return None

def get_actions_to_propose_cheap(graph, env, state):
    """Returns the actions to propose to reach the goal.

    This performs a set difference between the valid actions and the actions already taken in the graph
    to ensure that the same action is not proposed twice; however, actions may lead to states that have
    already been visited.

    Parameters:
        graph (nx.DiGraph)
            The graph to propose actions in.
        env (gym.Env)
            The environment to propose actions in.
        state (object)
            The current state of the environment.
    
    Returns:
        actions_to_propose (list)
            The actions to propose to reach the goal.
    """
    valid_actions = pddlgym_utils.get_valid_actions(env, state)
    actions_taken = [graph[hash(state)][node]["action"] for node in graph.successors(hash(state))]
    return list(set(valid_actions) - set(actions_taken))

def get_actions_to_propose(graph, env, state):
    """Returns the actions to propose to reach the goal.

    This function is similar to get_actions_to_propose_cheap, but it checks if the next state has already
    been visited in the graph before proposing the action.

    Parameters:
        graph (nx.DiGraph)
            The graph to propose actions in.
        env (gym.Env)
            The environment to propose actions in.
        state (object)
            The current state of the environment.
    
    Returns:
        actions_to_propose (list)
            The actions to propose to reach the goal.
    """
    valid_actions = pddlgym_utils.get_valid_actions(env, state)
    actions_to_propose = []
    for action in valid_actions:
        env_copy = deepcopy(env)
        next_state, _, _, _, _ = env_copy.step(action)
        if hash(next_state) not in graph.nodes:
            actions_to_propose.append(action)
    return actions_to_propose

def propose_actions(graph, env, state, plan, num_actions=1):
    """Proposes an action(s) to take in order to reach the goal.

    This performs a set difference between the valid actions and the actions already taken in the graph
    to ensure that the same action is not proposed twice.

    Parameters:
        graph (nx.DiGraph)
            The graph to propose actions in.
        env (gym.Env)
            The environment to propose actions in.
        state (object)
            The current state of the environment.
        plan (object)
            The plan to use to propose actions.
        num_actions (int)
            The number of actions to propose.
    
    Returns:
        actions (list)
            The action(s) to take in order to reach the goal.
    """
    # TODO: Use LLM
    actions_to_propose = get_actions_to_propose(graph, env, state)
    return random.sample(actions_to_propose, k=num_actions)

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
        img_path = pddlgym_utils.get_image_path(env_copy)
        graph.add_node(hash(next_state), label="", image=img_path, state=next_state, env=env_copy)
        graph.add_edge(hash(current_state), hash(next_state), label=str(action), action=action)

def select_state(graph, plan, goal):
    """Selects the next state to propose actions from.

    Parameters:
        graph (nx.DiGraph)
            The graph to select the next state from.
        plan (object)
            The plan to use to select the next state.
        goal (object)
            The goal to reach.
    
    Returns:
        selected_state (object)
            The next state to propose actions from.
    
    Raises:
        AssertionError
            There are no states left to propose actions from. This should never happen
            since the goal should be reached before this point.
    """
    # TODO: Use LLM
    sampled_nodes = random.sample(graph.nodes, k=len(graph.nodes))
    for node in sampled_nodes:
        state = graph.nodes[node]['state']
        env = graph.nodes[node]['env']
        if pddlgym_utils.did_reach_goal(state, goal) or len(get_actions_to_propose(graph, env, state)) > 0:
            # A goal state is in the graph or there are still actions left to propose
            return state
    assert False, "No states left to propose actions from."

def style_goal_nodes(graph, current_state, next_state):
    """Styles the goal nodes and edges in the graph.
    
    Parameters:
        graph (nx.DiGraph)
            The graph to style the goal nodes and edges in.
        current_state (object)
            The current state of the environment.
        next_state (object)
            The next state of the environment.
    
    Side Effects:
        Modifies the graph by styling the goal nodes and edges.
    """
    graph.nodes[current_state]["color"] = "red"
    graph.nodes[current_state]["penwidth"] = "6"
    graph.nodes[current_state]["group"] = "goal"
    graph.nodes[next_state]["color"] = "red"
    graph.nodes[next_state]["penwidth"] = "6"
    graph.nodes[next_state]["group"] = "goal"
    graph[current_state][next_state]["color"] = "red"
    graph[current_state][next_state]["penwidth"] = "6"

def plan(env, initial_state, goal, max_steps=20):
    """Follows the V0 single heuristic planning algorithm to output a sequence of actions to the goal."""

    # Generate plan
    plan = generate_plan()

    # Follow plan to reach goal
    graph = nx.DiGraph()
    img_path = pddlgym_utils.get_image_path(env)
    graph.add_node(hash(initial_state), label="", image=img_path, state=initial_state, env=deepcopy(env))
    selected_state = initial_state
    steps = 0
    pbar = tqdm(total=max_steps)
    while not pddlgym_utils.did_reach_goal(selected_state, goal) and steps < max_steps:
        curr_env = graph.nodes[hash(selected_state)]["env"]
        # Propose actions
        actions = propose_actions(graph, curr_env, selected_state, plan)
        # Compute the next states to add as nodes in the graph with directed action edges from the current state
        compute_next_states(graph, curr_env, selected_state, actions)
        # Select next state
        selected_state = select_state(graph, plan, goal)
        steps += 1
        pbar.update(1)
    
    # Get the shortest action sequence to the goal
    shortest_path = nx.shortest_path(graph, hash(initial_state), hash(selected_state))
    action_sequence = []
    for i in range(len(shortest_path) - 1):
        current_state = shortest_path[i]
        next_state = shortest_path[i + 1]
        action = graph[current_state][next_state]["action"]
        action_sequence.append(action)
        style_goal_nodes(graph, current_state, next_state)
    return action_sequence, graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", required=True, help="The name of the environment.")
    parser.add_argument("--graph_file", required=True, help="The name of the file to save the graph to.")
    parser.add_argument("--max_steps", type=int, default=20, help="The maximum number of steps to take to reach the goal.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed to use.")
    args = parser.parse_args()
    env_name = f"PDDLEnv{args.env_name.capitalize()}-v0"
    env = pddlgym_utils.make_pddlgym_env(env_name)
    random.seed(args.seed)
    initial_state, _ = env.reset()
    goal = initial_state.goal
    action_sequence, graph = plan(env, initial_state, goal, max_steps=args.max_steps)

    # Draw graph
    pygraphviz_graph = nx.nx_agraph.to_agraph(graph)
    pygraphviz_graph.layout('dot')
    pygraphviz_graph.draw(args.graph_file)