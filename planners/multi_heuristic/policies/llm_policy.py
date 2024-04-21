"""
This module contains the LLMPolicy class. The LLM policy plan generation, action proposal,
and state selection can be set to a ground truth mode which allows the user to interactively
input this information. This is intended for ablation studies and debugging purposes.

TODO(chalo2000): Allow for interactive mode and actual ground truth mode calculated by optimal
policy.
"""
import os
from copy import deepcopy
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from .policy import PlanPolicy

class LLMPolicy(PlanPolicy):
    """A plan policy that queries an LLM to propose actions and select next states."""
        
    def __init__(self, kwargs):
        """Initializes the LLM policy.
        
        Parameters:
            kwargs (dict)
                prompt_fn (function)
                    The function to use to prompt the LLM.
                llm (dict)
                    The Hydra configurations for the LLM policy.
                planner (dict)
                    The Hydra configurations for the planner.
        """
        super().__init__(kwargs)
        self.prompt_fn = kwargs["prompt_fn"]
        self.log_file = kwargs["planner"].get("log_file", None)
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # State translation
        self.state_translation_prompt_params = kwargs["llm"].get("state_translation_prompt", {})

        # Plan generation
        self.ground_truth_plan = kwargs["llm"].get("ground_truth_plan", False)
        self.plan_generation_prompt_params = kwargs["llm"].get("plan_generation_prompt", {})

        # Action proposal
        self.ground_truth_action = kwargs["llm"].get("ground_truth_action", False)
        self.action_proposal_prompt_params = kwargs["llm"].get("action_proposal_prompt", {})

        # State selection
        self.ground_truth_state_selection = kwargs["llm"].get("ground_truth_state_selection", False)
        self.state_selection_prompt_params = kwargs["llm"].get("state_selection_prompt", {})

        self.done = False
    
    def is_done(self):
        """Returns whether the policy is done.
        
        The LLM policy is done whenever the selected state satisfies the goal,
        checked by the environment model.
        
        Returns:
            done (bool)
                Whether the policy is done.
        """
        return self.done
    
    def _write_to_log(self, log_file, data):
        """Writes data to a log file.
        
        Parameters:
            log_file (str)
                The name of the log file to write to.
            data (str)
                The data to write to the log file.
        """
        with open(log_file, "a") as f:
            f.write(data)
    
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
            plan = input().replace("\\n", "\n")
            if self.log_file:
                log = f"Plan Generation (GT)\n{'-'*20}\nPlan:\n{plan}\n\n"
                self._write_to_log(self.log_file, log)
            return plan
        return None # TODO(chalo2000): Add LLM plan generation
    
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
            # TODO(chalo2000): Calculate ground truth with Dijkstra's algorithm
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
        if self.log_file:
                log = f"Action Proposal\n{'-'*20}\n{action_proposal_prompt}\n{action_proposal_response}\n\n"
                self._write_to_log(self.log_file, log)
        return matching_action
    
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
        model = graph.nodes[selected_node]["model"]
        self.done = model.did_reach_goal(selected_state, goal)
        if self.log_file:
                log = f"State Selection (GT)\n{'-'*20}\nState Index: {input_state}\n\n"
                self._write_to_log(self.log_file, log)
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
            # TODO(chalo2000): Calculate ground truth with Dijkstra's algorithm
            return self._interactive_select_state(graph, plan, goal)
        raise NotImplementedError # TODO(chalo2000): Add LLM state selection