"""
This module contains utility functions that can be shared across different policies.
"""
from copy import deepcopy

def get_actions_to_propose_cheap(graph, model, state):
    """Returns the actions to propose to reach the goal.

    This performs a set difference between the valid actions and the actions already taken in the graph
    to ensure that the same action is not proposed twice; however, actions may lead to states that have
    already been visited.

    Parameters:
        graph (nx.DiGraph)
            The graph to propose actions in.
        model (Model)
            The model to propose actions with.
        state (object)
            The current state of the environment.
    
    Returns:
        actions_to_propose (list)
            The actions to propose to reach the goal.
    """
    valid_actions = model.get_valid_actions(state)
    actions_taken = [graph[hash(state)][node]["action"] for node in graph.successors(hash(state))]
    return list(set(valid_actions) - set(actions_taken))

def get_actions_to_propose(graph, model, state):
    """Returns the actions to propose to reach the goal.

    This function is similar to get_actions_to_propose_cheap, but it checks if the next state has already
    been visited in the graph before proposing the action.

    Parameters:
        graph (nx.DiGraph)
            The graph to propose actions in.
        model (Model)
            The model to propose actions with.
        state (object)
            The current state of the environment.
    
    Returns:
        actions_to_propose (list)
            The actions to propose to reach the goal.
    """
    valid_actions = model.get_valid_actions(state)
    actions_to_propose = []
    for action in valid_actions:
        model_copy = deepcopy(model)
        next_state, _, _, _, _ = model_copy.env.step(action)
        if hash(next_state) not in graph.nodes:
            actions_to_propose.append(action)
    return actions_to_propose