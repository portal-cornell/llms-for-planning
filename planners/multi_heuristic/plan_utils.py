from copy import deepcopy
import networkx as nx
from tqdm import tqdm
import pddlgym_utils

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
        graph.add_node(hash(next_state), state=next_state, env=env_copy)
        graph.add_edge(hash(current_state), hash(next_state), action=action)

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

def visualize_graph(graph, graph_file):
    """Visualizes the graph and saves it to a file.
    
    Parameters:
        graph (nx.DiGraph)
            The graph to visualize.
        graph_file (str)
            The name of the file to save the graph to.

    Side Effects:
        - Creates temporary image files
        - Modifies the graph's labels and images
        - Saves the graph to a file
    """
    for node in graph.nodes:
        graph.nodes[node]["label"] = ""
        graph.nodes[node]["image"] = pddlgym_utils.get_image_path(graph.nodes[node]["env"])
    for edge in graph.edges:
        graph[edge[0]][edge[1]]["label"] = str(graph[edge[0]][edge[1]]["action"])
    pygraphviz_graph = nx.nx_agraph.to_agraph(graph)
    pygraphviz_graph.layout('dot')
    pygraphviz_graph.draw(graph_file)

def plan(plan_policy, env, initial_state, goal, max_steps=20):
    """Follows the V0 single heuristic planning algorithm to output a sequence of actions to the goal.
    
    Parameters:
        plan_policy (object)
            The plan policy to use to generate a plan, propose actions, and select the next state.
        env (gym.Env)
            The environment to plan in.
        initial_state (object)
            The initial state of the environment.
        goal (object)
            The goal to reach.
        max_steps (int)
            The maximum number of steps to take to reach the goal.
    
    Returns:
        action_sequence (list)
            The sequence of actions to take to reach the goal.
        graph (nx.DiGraph)
            The graph of states and actions taken to reach the goal.
    """

    # Generate plan
    plan = plan_policy.generate_plan()

    # Follow plan to reach goal
    graph = nx.DiGraph()
    graph.add_node(hash(initial_state), state=initial_state, env=deepcopy(env))
    selected_state = initial_state
    steps = 0
    pbar = tqdm(total=max_steps)
    while steps < max_steps:
        curr_env = graph.nodes[hash(selected_state)]["env"]
        # Propose actions
        actions = plan_policy.propose_actions(graph, curr_env, selected_state, plan,)
        # # Compute the next states to add as nodes in the graph with directed action edges from the current state
        compute_next_states(graph, curr_env, selected_state, actions)
        # # Select next state
        selected_state = plan_policy.select_state(graph, plan, goal)
        steps += 1
        pbar.update(1)
    
    # # Get the shortest action sequence to the goal
    shortest_path = nx.shortest_path(graph, hash(initial_state), hash(selected_state))
    action_sequence = []
    for i in range(len(shortest_path) - 1):
        current_state = shortest_path[i]
        next_state = shortest_path[i + 1]
        action = graph[current_state][next_state]["action"]
        action_sequence.append(action)
        style_goal_nodes(graph, current_state, next_state)
    return action_sequence, graph