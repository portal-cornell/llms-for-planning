"""This module contains the PlanPolicy class. All plan policies should inherit from this class."""

class PlanPolicy:

    def __init__(self, kwargs):
        """Initializes the plan policy.
        
        Parameters:
            kwargs (dict)
                The keyword arguments for the policy.
        """
        self.kwargs = kwargs

    def generate_plan(self):
        """Generates a plan to reach the goal.
        
        Raises:
            NotImplementedError
                This function should be implemented in a subclass.
        """
        raise NotImplementedError # TODO: Figure out parameters with LLM

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
                The plan to use to propose actions.
        
        Raises:
            NotImplementedError
                This function should be implemented in a subclass.
        """
        raise NotImplementedError
    
    def select_state(self, graph, plan, goal):
        """Selects the next state to propose actions from.
        
        Parameters:
            graph (nx.DiGraph)
                The graph to select the next state from.
            plan (object)
                The plan to use to select the next state.
            goal (object)
                The goal to reach.
        
        Raises:
            NotImplementedError
                This function should be implemented in a subclass.
        """
        raise NotImplementedError