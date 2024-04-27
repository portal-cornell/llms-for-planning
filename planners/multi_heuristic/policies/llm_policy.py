"""
This module contains the LLMPolicy class. The LLM policy plan generation, action proposal,
and state selection can be set to a ground truth mode which allows the user to interactively
input this information. This is intended for ablation studies and debugging purposes.

TODO(chalo2000): Allow for interactive mode and actual ground truth mode calculated by optimal
policy.
"""
import os
import openai
import re
from copy import deepcopy
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from .policy import PlanPolicy
from . import utils

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
        self.state_descriptions = {} # Cache for state descriptions

        # State translation
        self.state_translation_prompt_params = kwargs["llm"]["prompts"].get("state_translation_prompt", {})

        # Plan generation
        self.ground_truth_plan = False
        self.plan_generation_prompt_params = kwargs["llm"]["prompts"].get("plan_generation_prompt", {})

        # Action proposal
        self.ground_truth_action = False
        self.action_proposal_prompt_params = kwargs["llm"]["prompts"].get("action_proposal_prompt", {})

        # State selection
        self.ground_truth_state_selection = False
        self.state_selection_prompt_params = kwargs["llm"]["prompts"].get("state_selection_prompt", {})

        # Planner params
        self.cheap = kwargs["planner"].get("cheap", True)

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
            f.write(data + "\n\n")
    
    def _prompt_llm(self, user_prompt, params, history=[]):
        """Prompts the LLM with messages and parameters.
        
        Parameters:
            user_prompt (str)
                The user prompt to query the LLM with.
            params (dict)
                The parameters to prompt the LLM with.
            history (list)
                The history of the conversation.
        
        Returns:
            response (str)
                The response from the LLM.
            truncated_history (list)
                The truncated history that fits within the context length.
        """
        success = False
        truncated_history = history
        while not success:
            try:
                response = self.prompt_fn(user_prompt, **params, history=truncated_history)
                success = True
            except openai.BadRequestError as e:
                error_code = e.code
                if error_code == 'context_length_exceeded':
                    raise NotImplementedError("Context length exceeded. Please implement truncation.")
                else:
                    raise e # Raise other errors for user to handle
        return response, truncated_history
    
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
        return goal
    
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
        if self.ground_truth_action:
            # TODO(chalo2000): Calculate ground truth with Dijkstra's algorithm
            return self._interactive_propose_actions(graph, model, state, plan)
        
        # Prepare action proposal user prompt

        # Get goal state description
        if self.state_descriptions.get(hash(plan)):
            goal_description = self.state_descriptions[hash(plan)]
        else:
            goal_str = model.goal_to_str(plan) # Plan is the goal
            goal_description, _ = self._prompt_llm(goal_str, self.state_translation_prompt_params)
            self.state_descriptions[hash(plan)] = goal_description
        
        # Get current node state description
        if self.state_descriptions.get(hash(state)):
            state_description = self.state_descriptions[hash(state)] # Use cached state description
        else:
            state_str = model.state_to_str(state)
            state_description, _ = self._prompt_llm(state_str, self.state_translation_prompt_params)

        # Get valid actions left to propose
        valid_actions = self._actions_to_propose(graph, model, state)
        valid_actions_str = "\n".join([f"- {action}" for action in valid_actions])

        # Get action from action proposal response
        action_proposal_prompt =  f"Goal:\n{goal_description}\n"
        action_proposal_prompt += f"Current state:\n{state_description}\n"
        action_proposal_prompt += f"Valid Actions:\n{valid_actions_str}\n"
        self._write_to_log(self.log_file, action_proposal_prompt)
        action_proposal_response, _ = self._prompt_llm(action_proposal_prompt, self.action_proposal_prompt_params)
        self._write_to_log(self.log_file, action_proposal_response)

        # Extract and return action
        regex = r"Action:\s*(.+)"
        match = re.search(regex, action_proposal_response)
        if not match:
            self.done = True # Malformed response; kill the planner
            return []
        action = match.group(1)
        action = action.replace(" ", "") # Remove spaces
        valid_actions = model.get_valid_actions(state)
        matching_action = list(filter(lambda x: str(x) == action, valid_actions))
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
        if len(actions) == 0:
            return # No valid action found
        
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

        goal_description = self.state_descriptions[hash(plan)] # Plan is the goal
        states_to_select = []
        for i, node in enumerate(graph.nodes):
            state = graph.nodes[node]["state"]
            model = graph.nodes[node]["model"]
            if self.state_descriptions.get(hash(state)):
                state_description = self.state_descriptions[hash(state)]
            else:
                state_str = model.state_to_str(state)
                state_description, _ = self._prompt_llm(state_str, self.state_translation_prompt_params)
                self.state_descriptions[hash(state)] = state_description
            
            if len(self._actions_to_propose(graph, model, state)) > 0:
                # Only append states that have valid actions to propose to save on LLM context length
                states_to_select.append(f"s{i}:\n{state_description}")
        
        # Get state from state selection response
        state_selection_prompt = f"Goal:\n{goal_description}\n"
        state_selection_prompt += f"States:\n"
        state_selection_prompt += "\n".join(states_to_select)
        self._write_to_log(self.log_file, state_selection_prompt)
        state_selection_response, _ = self._prompt_llm(state_selection_prompt, self.state_selection_prompt_params)
        self._write_to_log(self.log_file, state_selection_response)

        # Extract and return state
        regex = r"Choice:\s*s(\d+)"
        match = re.search(regex, state_selection_response)
        if not match:
            self.done = True # Malformed response; kill the planner
            return []
        state_choice = match.group(1)
        selected_node = list(graph.nodes)[int(state_choice)]
        selected_state = graph.nodes[selected_node]["state"]
        
        # Check if selected state is goal
        model = graph.nodes[selected_node]["model"]
        self.done = model.did_reach_goal(selected_state, goal)

        return selected_state

