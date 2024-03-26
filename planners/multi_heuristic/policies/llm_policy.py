from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import random

from .policy import PlanPolicy
from . import utils

class LLMPolicy(PlanPolicy):
    """A plan policy that queries an LLM to propose actions and select next states."""
        
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
        self.prompt_fn = kwargs["prompt_fn"]
        
        # State translation
        self.state_translation_prompt_params = kwargs.get("state_translation_prompt", {})

        # Plan generation
        self.ground_truth_plan = kwargs.get("ground_truth_plan", False)
        self.plan_generation_prompt_params = kwargs.get("plan_generation_prompt", {})

        # Action proposal
        self.ground_truth_action = kwargs.get("ground_truth_action", False)
        self.action_proposal_prompt_params = kwargs.get("action_proposal_prompt", {})

        # State selection
        self.ground_truth_state_selection = kwargs.get("ground_truth_state_selection", False)
        self.state_selection_prompt_params = kwargs.get("state_selection_prompt", {})
    
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
        if self.ground_truth_plan:
            print("Generate a plan:")
            print(model.state_to_str(initial_state))
            print(f"Goal: {goal}")
            return input() # TODO: Add ground truth per environment
        return None # TODO: Add LLM plan generation
    
    def _interactive_graph_visualize(self, graph, state=None):
        """Displays a numbered graph to assist with interactions.

        Parameters:
            graph (nx.DiGraph)
                The graph to visualize.
            state (object)
                The current state of the environment (to highlight in the graph).
        
        Side Effects:
            - Displays the graph
        """
        for i, node in enumerate(graph.nodes):
            graph.nodes[node]["fontsize"] = "60"
            graph.nodes[node]["label"] = str(i)
            model = graph.nodes[node]["model"]
            graph.nodes[node]["image"] = model.get_image_path()
            if state == graph.nodes[node]["state"]:
                graph.nodes[node]["color"] = "red"
                graph.nodes[node]["penwidth"] = "6"
            else:
                graph.nodes[node]["color"] = "black"
                graph.nodes[node]["penwidth"] = "1"
        for edge in graph.edges:
            graph[edge[0]][edge[1]]["label"] = str(graph[edge[0]][edge[1]]["action"])
        pygraphviz_graph = nx.nx_agraph.to_agraph(graph)
        pygraphviz_graph.layout('dot')
        # Display graph
        img = Image.open(BytesIO(pygraphviz_graph.draw(format='png')))
        plt.close()
        plt.imshow(img)
        plt.show(block=False)
    
    def _interactive_propose_actions(self,graph, model, state, plan):
        """Proposes actions interactively to take in order to reach the goal.
        
        Parameters:
            graph (nx.DiGraph)
                The graph to propose actions in.
            model (Model)
                The model to propose actions with.
            state (object)
                The current state of the environment.
            plan (object)
                The plan to use to propose actions.
        
        Returns:
            actions (list)
                The actions to take to reach the goal.
        """
        print("Choose a valid action to propose:")
        print(model.state_to_str(state))
        valid_actions = model.get_valid_actions(state)
        for i, action in enumerate(valid_actions):
            print(f"{i}: {action}")
        print(f"Plan: {plan}")
        input_action = ""
        while not input_action.isdigit() or int(input_action) >= len(valid_actions):
            self._interactive_graph_visualize(graph, state) # Display graph
            input_action = input("Enter action idx: ")
        action = valid_actions[int(input_action)]
        print(action)
        return [action]
    
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
        if self.ground_truth_action:
            # TODO: Calculate ground truth with Dijkstra's algorithm
            print("ground truth")
            return self._interactive_propose_actions(graph, model, state, plan)
        
        # Get initial node state description
        root_node = list(graph.nodes)[0]
        initial_state = graph.nodes[root_node]["state"]
        initial_state_str = model.state_to_str(initial_state)
        initial_state_description = self.prompt_fn(initial_state_str, **self.state_translation_prompt_params)
        
        # Get current node state description
        state_str = model.state_to_str(state)
        state_description = self.prompt_fn(state_str, **self.state_translation_prompt_params)

        # Get action from action proposal response
        action_proposal_prompt =  f"Initial state:\n{initial_state_description}\n"
        action_proposal_prompt += f"Plan:\n{plan}\n"
        action_proposal_prompt += f"Current state:\n{state_description}"
        print(action_proposal_prompt)
        action_proposal_response = self.prompt_fn(action_proposal_prompt, **self.action_proposal_prompt_params)
        print(action_proposal_response)
        action = action_proposal_response.split("Action:")[1].strip()
        # Filter valid actions and cast to string until finding the action
        matching_action = list(filter(lambda x: str(x) == action, model.get_valid_actions(state)))
        print(matching_action)
        return matching_action #self._interactive_propose_actions(graph, model, state, plan) #raise NotImplementedError # TODO: Add LLM action proposal
    
    def _interactive_select_state(self, graph, plan, goal):
        """Selects the next state to propose actions from interactively.
        
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
        """
        print("Select a state:")
        print(f"Plan: {plan}")
        input_state = ""
        while not input_state.isdigit() or int(input_state) >= len(graph.nodes):
            self._interactive_graph_visualize(graph) # Display graph
            input_state = input("Enter state idx: ")
        selected_node = list(graph.nodes)[int(input_state)]
        selected_state = graph.nodes[selected_node]["state"]
        return selected_state

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
        if self.ground_truth_state_selection:
            # TODO: Calculate ground truth with Dijkstra's algorithm
            return self._interactive_select_state(graph, plan, goal)
        raise NotImplementedError # TODO: Add LLM state selection