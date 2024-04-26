"""
This module contains the RandomPolicy class. It has no notion of generating a plan and instead
randomly proposes actions and selects next states. This policy does not use an LLM.
"""
import random
from copy import deepcopy

from .policy import PlanPolicy
from . import utils

class RandomPolicy(PlanPolicy):
    """A plan policy that randomly proposes actions and selects next states."""
        
    def __init__(self, kwargs):
        """Initializes the random policy.
        
        Parameters:
            kwargs (dict)
                The keyword arguments for the policy which include:
                cheap (bool)
                    Whether to use the cheap version of the get_actions_to_propose function.
                num_actions (int)
                    The number of actions to propose.

        """
        super().__init__(kwargs)

        self.cheap = kwargs['planner'].get("cheap", False)
        self.num_actions = kwargs['planner'].get("num_actions", 1)
        self.done = False
    
    def is_done(self):
        """Returns whether the policy is done.
        
        The random policy is done when the Gym environment reaches the goal.

        Returns:
            done (bool)
                Whether the policy is done.
        """
        return self.done
    
    def generate_plan(self, model, initial_state, goal):
        """Generates a plan to reach the goal.
        
        Parameters:
            model (Model)
                The model to translate state with.
            initial_state (object)
                The initial state of the environment.
            goal (object)
                The goal to reach.
        
        Returns:
            None
                This policy does not generate a plan.
        """
        return None
    
    def _actions_to_propose(self, graph, model, state):
        """Returns the actions to propose to reach the goal.
        
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
        if self.cheap:
            return utils.get_actions_to_propose_cheap(graph, model, state)
        return utils.get_actions_to_propose(graph, model, state)

    def propose_actions(self, graph, model, state, plan):
        """Proposes an action(s) to take in order to reach the goal.
        
        Parameters:
            graph (nx.DiGraph)
                The graph to propose actions in.
            model (Model)
                The model to propose actions with.
            state (object)
                The current state of the environment.
            plan (object)
                The plan to use to propose actions. This is not used in this policy.
        
        Raises:
            NotImplementedError
                This function should be implemented in a subclass.
        """
        actions_to_propose = self._actions_to_propose(graph, model, state)
        return random.sample(actions_to_propose, k=min(self.num_actions, len(actions_to_propose)))
    
    def compute_next_states(self, graph, model, current_state, actions):
        """Computes the next states and updates the graph.

        Parameters:
            graph (nx.DiGraph)
                The graph to add the next states to.
            model (Model)
                The model containing the environment to simulate the actions in.
            current_state (object)
                The current state of the environment.
            actions (list)
                The actions to simulate in the environment.
        
        Side Effects:
            Modifies the graph by adding the next states as nodes and the actions as edges.
        """
        for action in actions:
            model_copy = deepcopy(model)
            next_state, _, _, _, _ = model_copy.env.step(action)
            graph.add_node(hash(next_state), state=next_state, model=model_copy)
            graph.add_edge(hash(current_state), hash(next_state), action=action)
    
    def select_state(self, graph, plan, goal):
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
        sampled_nodes = random.sample(graph.nodes, k=len(graph.nodes))
        for node in sampled_nodes:
            state = graph.nodes[node]['state']
            model = graph.nodes[node]['model']
            
            reached_goal = model.did_reach_goal(state, goal)
            if reached_goal or len(self._actions_to_propose(graph, model, state)) > 0:
                # A goal state is in the graph or there are still actions left to propose
                self.done = reached_goal
                return state
        assert False, "No states left to propose actions from."