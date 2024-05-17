"""
This module contains the V0 single heuristic planner which uses an LLM to generate a single plan 
to reach a goal in a given environment. Then, starting from an initial state, the plan is used to
propose an action(s) to take in order to reach the goal. Using a model of the environment, the 
next states are computed and the plan is used to select the next state to propose actions from. 
This process is repeated until the goal is reached.

The V0 single heuristic planner is the simplest planner that we'll use to test the effectiveness
of an LLM as an action proposer and state selector.
"""
import os
from copy import deepcopy
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
from tqdm import tqdm

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

def visualize_graph(graph, graph_file="", number_nodes=False):
    """Visualizes the graph by saving to file or displaying.
    
    Parameters:
        graph (nx.DiGraph)
            The graph to visualize.
        graph_file (str)
            The name of the file to save the graph to. If empty, the graph is displayed.
        number_nodes (bool)
            Whether to number the nodes in the graph.

    Side Effects:
        - Creates temporary image files
        - Modifies the graph's labels and images
        - Saves the graph to a file
    """
    for i, node in enumerate(graph.nodes):
        graph.nodes[node]["fontsize"] = "60"
        graph.nodes[node]["label"] = str(i) if number_nodes else ""
        model = graph.nodes[node]["model"]
        graph.nodes[node]["image"] = model.get_image_path()
    for edge in graph.edges:
        graph[edge[0]][edge[1]]["label"] = str(graph[edge[0]][edge[1]]["action"])
    pygraphviz_graph = nx.nx_agraph.to_agraph(graph)
    pygraphviz_graph.layout('dot')
    if graph_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(graph_file), exist_ok=True)
        pygraphviz_graph.draw(graph_file)
    else:
        img = Image.open(BytesIO(pygraphviz_graph.draw(format='png')))
        plt.imshow(img)
        plt.show()

def plan(plan_policy, model, initial_state, goal, max_steps=20):
    """Follows the V0 single heuristic planning algorithm to output a sequence of actions to the goal.
    
    Parameters:
        plan_policy (object)
            The plan policy to use to generate a plan, propose actions, and select the next state.
        model (Model)
            The model to query for environment interaction, valid actions, and goal satisfaction.
        initial_state (object)
            The initial state of the environment.
        goal (object)
            The goal to reach.
        max_steps (int)
            The maximum number of steps to take to reach the goal.
    
    Returns:
        reached_goal (bool)
            Whether the goal was reached.
        action_sequence (list)
            The sequence of actions to take to reach the goal.
        graph (nx.DiGraph)
            The graph of states and actions taken to reach the goal.
    """

    # Generate plan
    plan = plan_policy.generate_plan(model, initial_state, goal)

    # Follow plan to reach goal
    graph = nx.DiGraph()
    graph.add_node(hash(initial_state), state=initial_state, model=deepcopy(model))
    selected_state = initial_state
    steps = 0
    pbar = tqdm(total=max_steps)
    while not plan_policy.is_done() and steps < max_steps:
        curr_model = graph.nodes[hash(selected_state)]["model"]
        # Propose actions
        actions = plan_policy.propose_actions(graph, curr_model, selected_state, plan, 'sokoban')
        if type(actions) == dict:
            # For state+action proposal
            selected_state = actions["state"]
            actions = actions["actions"]
        # Compute the next states to add as nodes in the graph with directed action edges from the current state
        plan_policy.compute_next_states(graph, curr_model, selected_state, actions, 'sokoban')
        # Select next state
        selected_state = plan_policy.select_state(graph, plan, goal)
        steps += 1
        pbar.update(1)
    
    # Get the shortest action sequence to the last selected state
    reached_goal = model.did_reach_goal(selected_state, goal)
    shortest_path = nx.shortest_path(graph, hash(initial_state), hash(selected_state))
    action_sequence = []
    for i in range(len(shortest_path) - 1):
        current_state = shortest_path[i]
        next_state = shortest_path[i + 1]
        action = graph[current_state][next_state]["action"]
        action_sequence.append(action)
        if reached_goal:
            style_goal_nodes(graph, current_state, next_state)
    return reached_goal, action_sequence, graph