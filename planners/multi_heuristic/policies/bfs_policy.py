import random

from policies.policy import PlanPolicy
import pddlgym_utils
import policies.utils as utils

class BFSPolicy(PlanPolicy):
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
        self.cheap = kwargs.get("cheap", False)
        self.num_actions = kwargs.get("num_actions", 1)
    
    def generate_plan(self):
        """Generates a plan to reach the goal.
        
        Returns:
            None
                This policy does not generate a plan.
        """
        return None
    
    def _actions_to_propose(self, graph, env, state):
        """Returns the actions to propose to reach the goal.
        
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
        if self.cheap:
            return utils.get_actions_to_propose_cheap(graph, env, state)
        return utils.get_actions_to_propose(graph, env, state)

    def propose_actions(self, graph, env, state, plan):
        """Proposes an action(s) to take in order to reach the goal.
        
        Parameters:
            graph (nx.DiGraph)
                The graph to propose actions in.
            env (gym.Env)
                The environment to propose actions in.
            state (object)
                The current state of the environment.
            plan (object)
                The plan to use to propose actions. This is not used in this policy.
        
        Raises:
            NotImplementedError
                This function should be implemented in a subclass.
        """
        actions_to_propose = self._actions_to_propose(graph, env, state)
        return random.sample(actions_to_propose, k=min(self.num_actions, len(actions_to_propose)))
    
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
            env = graph.nodes[node]['env']

            if pddlgym_utils.did_reach_goal(state, goal) or len(self._actions_to_propose(graph, env, state)) > 0:
                # A goal state is in the graph or there are still actions left to propose
                return state
        assert False, "No states left to propose actions from."