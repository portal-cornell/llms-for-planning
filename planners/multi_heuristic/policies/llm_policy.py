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
import json
from copy import deepcopy
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from .policy import PlanPolicy
from . import utils
from .utils import convert_states_to_bitmap_sokoban, map_llm_action_sokoban

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
        self.action_feedback_msg = ""
        self.action_history = []

        # State selection
        self.ground_truth_state_selection = False
        self.state_selection_prompt_params = kwargs["llm"]["prompts"].get("state_selection_prompt", {})
        self.tree_json = {}
        self.state_selection_feedback_msg = ""
        self.current_state = None
        self.next_state = None
        self.state_history = []
        self.action_no_reasoning_history = []
        self.reflections = []
        self.current_states = []
        self.actions = []
        self.next_states = []

        # Planner params
        self.cheap = kwargs["planner"].get("cheap", True)

        self.chat_history = []
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
        self.current_state = initial_state
        self.next_state = initial_state
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
    
    def _get_state_id(self, graph, state):
        """Returns the ID of the state in the graph.
        
        Parameters:
            graph (nx.DiGraph)
                The graph to get the state ID from.
            state (object)
                The state to get the ID of.
        
        Returns:
            state_id (int)
                The ID of the state in the graph.
        """
        for i, node in enumerate(graph.nodes):
            if graph.nodes[node]["state"] == state:
                return i
        assert False, "State not found in graph"
    
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
        # if self.state_descriptions.get(hash(plan)):
        #     goal_description = self.state_descriptions[hash(plan)]
        # else:
        #     goal_str = model.goal_to_str(plan) # Plan is the goal
        #     goal_description, _ = self._prompt_llm(goal_str, self.state_translation_prompt_params)
        #     self.state_descriptions[hash(plan)] = goal_description
        goal_description = model.goal_to_str(state, plan) # Plan is the goal
        pretty_goal_description = json.dumps(goal_description, indent=4)

        # Get current node state description
        if self.state_descriptions.get(hash(state)):
            state_description = self.state_descriptions[hash(state)] # Use cached state description
        else:
            state_str = model.state_to_str(state)
            state_description, _ = self._prompt_llm(state_str, self.state_translation_prompt_params)
            self.state_descriptions[hash(state)] = state_description
        state_id = self._get_state_id(graph, state)
        intervention_msg = "" if self.current_state == self.next_state else " (Intervention)"

        # Get valid actions from model
        valid_actions = model.get_valid_actions(state) # self._actions_to_propose(graph, model, state)
        if len(valid_actions) == 0:
            self.state_selection_feedback_msg = f"There are no valid actions left to propose at State {state_id}. Please select a new state."
            return []
        valid_actions_str = "\n".join([f"- {action}" for action in valid_actions])

        # Get action from action proposal response
        action_proposal_prompt = ""
        if self.action_feedback_msg:
            action_proposal_prompt += f"Error Feedback: {self.action_feedback_msg}\n"
            self.action_feedback_msg = ""
        action_proposal_prompt +=  f"Goal Tracker:\n{pretty_goal_description}\n"
        action_proposal_prompt += f"Current State {state_id}{intervention_msg}:\n{state_description}\n"
        action_proposal_prompt += f"Valid Actions:\n{valid_actions_str}\n"
        
        # if len(valid_actions) == 1:
        #     self._write_to_log(self.log_file, action_proposal_prompt)
        #     self.chat_history.append(action_proposal_prompt)
        #     self.action_no_reasoning_history.append(action_proposal_prompt)
        #     skip_msg = f"[Skip LLM] Only one valid action: {valid_actions[0]}"
        #     self._write_to_log(self.log_file, skip_msg)
        #     self.chat_history.append(skip_msg)
        #     self.action_no_reasoning_history.append(skip_msg)
        #     return valid_actions
        action_proposal_response, self.action_history = self._prompt_llm(action_proposal_prompt, self.action_proposal_prompt_params, history=self.action_history)
        
        # Write to log and append to chat history
        self._write_to_log(self.log_file, "ACTION PROPOSAL PROMPT\n" + "-"*20)
        self._write_to_log(self.log_file, action_proposal_prompt)
        self.action_history.append(action_proposal_prompt)
        self._write_to_log(self.log_file, "ACTION PROPOSAL RESPONSE\n" + "-"*20)
        self._write_to_log(self.log_file, action_proposal_response)
        self.action_history.append(action_proposal_response)

        # Extract and return action
        regex = r"Action:\s*(.+)"
        match = re.search(regex, action_proposal_response)
        if not match:
            self.action_feedback_msg = "The action was malformed. Please provide a valid action in the form 'Action: <action>'.\n"
            # self.done = True # Malformed response; kill the planner
            return []
        action = match.group(1)
        if 'typed-sokoban.pddl' in model.env.env._domain_file:
            action = map_llm_action_sokoban(action)
        stripped_action = action.replace(" ", "") # Remove spaces
        matching_action = list(filter(lambda x: str(x) == stripped_action, valid_actions))
        if len(matching_action) == 0:
            self.action_feedback_msg = f"The action provided, '{action}' was invalid. Please provide a valid action from the list."
        self.action_no_reasoning_history.append(action_proposal_prompt)
        self.current_states.append(state)
        self.actions.append(action)

        # Add Reflect reasoning (if any)
        regex = r"Reflect:\s*(.+)"
        match = re.search(regex, action_proposal_response)
        if match:
            reflect = match.group(1)
            self.action_no_reasoning_history.append(f"Reflect: {reflect}\nAction: {action}")
            self.reflections.append(reflect)
        else:
            self.action_no_reasoning_history.append(f"Action: {action}")
            self.reflections.append("No reflection")
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
            # invalid_msg = "The action provided was invalid. Please provide a valid action from the list."
            # self._write_to_log(self.log_file, invalid_msg)
            # self.chat_history.append(invalid_msg)
            return # No valid action found
        
        for action in actions:
            model_copy = deepcopy(model)
            next_state, _, _, _, _ = model_copy.env.step(action)
            self.current_state = next_state
            self.next_state = next_state
            self.next_states.append(next_state)
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

    def _create_tree_json(self, graph):
        """Creates a JSON representation of the graph.
        
        The format of the JSON is
        {
            "steps": 0,
            "state": ...,
            "parent": ...,
            "action_to_parent": ...,
            "action_from_parent": ...,
            "actions_left": ...,
            "children": [
            {
                "steps": 1,
                "state": ...,
                "parent": ...,
                "action_to_parent": ...,
                "action_from_parent": ...,
                "actions_left": ...,
                "children": [...]
            },
            ...
            ]
        }

        Parameters:
            graph (nx.DiGraph)
                The graph to create a JSON representation of.
        
        Returns:
            tree_json (dict)
                The JSON representation of the graph.
        """
        # Create tree JSON
        tree_json = {}
        root = list(graph.nodes)[0]

        tree_json["steps"] = 0
        tree_json["state"] = "State 0"
        tree_json["parent"] = None
        tree_json["action_to_parent"] = None
        tree_json["action_from_parent"] = None
        str_actions_to_propose = [str(action) for action in self._actions_to_propose(graph, graph.nodes[root]["model"], graph.nodes[root]["state"])]
        tree_json["actions_left"] = str_actions_to_propose
        tree_json["children"] = []
        visited = set()
        visited.add(root)
        stack = [(root, tree_json)]
        while stack:
            current_node, current_json = stack.pop()
            for child in graph.successors(current_node):
                if child in visited:
                    continue
                visited.add(child)
                child_json = {}
                child_json["steps"] = current_json["steps"] + 1
                child_json["state"] = f"State {self._get_state_id(graph, graph.nodes[child]['state'])}"
                child_json["parent"] = current_json["state"]
                # Action to parent may have not been added yet so could be None
                action_to_parent_edge = graph.get_edge_data(child, current_node)
                child_json["action_to_parent"] = str(action_to_parent_edge["action"]) if action_to_parent_edge is not None else None
                child_json["action_from_parent"] = str(graph[current_node][child]["action"])
                str_actions_to_propose = [str(action) for action in self._actions_to_propose(graph, graph.nodes[child]["model"], graph.nodes[child]["state"])]
                child_json["actions_left"] = str_actions_to_propose
                child_json["children"] = []
                current_json["children"].append(child_json)
                stack.append((child, child_json))
        return tree_json
    
    def _get_pairwise_comparison(self, graph, goal, s1, s2):
        """Returns the pairwise comparison between two states.
        
        Parameters:
            graph (nx.DiGraph)
                The graph to select the next state from.
            goal (object)
                The goal to reach.
            s1 (object)
                The first state to compare.
            s2 (object)
                The second state to compare.
        
        Returns:
            comparison (bool)
                True if s1 >= s2, False otherwise.
        
        Side Effects:
            - Writes to the log file
        """
        goal_description = self.state_descriptions[hash(goal)]
        s1_description = self.state_descriptions[hash(s1)]
        s1_id = self._get_state_id(graph, s1)
        s2_description = self.state_descriptions[hash(s2)]
        s2_id = self._get_state_id(graph, s2)
        state_selection_prompt = f"Goal:\n{goal_description}\n"
        state_selection_prompt += f"State {s1_id}:\n{s1_description}\n"
        state_selection_prompt += f"State {s2_id}:\n{s2_description}\n"

        state_selection_response, self.chat_history = self._prompt_llm(state_selection_prompt, self.state_selection_prompt_params, history=self.chat_history)
        
        self._write_to_log(self.log_file, f"NOT INCLUDED IN HISTORY:\n{state_selection_prompt}")
        # self.chat_history.append(state_selection_prompt)
        self._write_to_log(self.log_file, f"NOT INCLUDED IN HISTORY:\n{state_selection_response}")
        # self.chat_history.append(state_selection_response)
        
        regex = r"Choice:\s*State\s*(\d+)"
        match = re.search(regex, state_selection_response)
        if not match:
            self.done = True # Malformed response; kill the planner
            return []
        state_choice = match.group(1)
        return state_choice == str(s1_id)

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
        
        action_history = ""
        # Generate state description list
        
        for state_hash, description in self.state_descriptions.items():
            state = graph.nodes[state_hash]["state"]
            model = graph.nodes[state_hash]["model"]
            state_id = self._get_state_id(graph, state)
            action_history += f"State {state_id}:\n{description}\n"
            valid_actions = model.get_valid_actions(state)
            valid_actions_str = "\n".join([f"- {action}" for action in valid_actions])
            action_history += f"Valid Actions:\n{valid_actions_str}\n\n"

            
        
        # Generate state transitions (reflection, curr_state, action, next_state)
        action_history += "State Transitions:\n"
        for reflection, curr_state, action, next_state in zip(self.reflections, self.current_states, self.actions, self.next_states):
            curr_state_id = self._get_state_id(graph, curr_state)
            next_state_id = self._get_state_id(graph, next_state)
            action_history += f"Reflection: {reflection} | State {curr_state_id} -> {action} -> State {next_state_id}\n"
        # Collect action history as user prompt
        # for i, chat in enumerate(self.action_no_reasoning_history):
        # # for i, chat in enumerate(self.action_history):
        #     role = "User" if i % 2 == 0 else "Assistant"
        #     action_history += f"{role}:\n{chat}\n\n"
        
        state_selection_prompt = ""
        if self.state_selection_feedback_msg:
            state_selection_prompt += f"Error Feedback: {self.state_selection_feedback_msg}\n"
            self.state_selection_feedback_msg = None
        # tree_json = self._create_tree_json(graph)
        # pretty_tree_json = json.dumps(tree_json, indent=4)
        # state_selection_prompt += f"State Space:\n{pretty_tree_json}\n"
        model = graph.nodes[hash(self.current_state)]["model"]
        if self.state_descriptions.get(hash(self.current_state)):
            state_description = self.state_descriptions[hash(self.current_state)] # Use cached state description
        else:
            state_str = model.state_to_str(self.current_state)
            state_description, _ = self._prompt_llm(state_str, self.state_translation_prompt_params)
            self.state_descriptions[hash(self.current_state)] = state_description
        state_id = self._get_state_id(graph, self.current_state)
        state_selection_prompt += f"Current State {state_id}:\n{state_description}\n"
        
        # Goal description
        goal_description = model.goal_to_str(self.current_state, plan) # Plan is the goal
        pretty_goal_description = json.dumps(goal_description, indent=4)
        
        state_selection_prompt += f"Goal Tracker:\n{pretty_goal_description}\n"
        user_prompt = f"{action_history}\n{state_selection_prompt}"
        state_selection_response, self.state_history = self._prompt_llm(user_prompt, self.state_selection_prompt_params, history=self.state_history)
        self._write_to_log(self.log_file, "STATE SELECTION PROMPT\n" + "-"*20)
        self._write_to_log(self.log_file, user_prompt)
        self.state_history.append(state_selection_prompt)
        self._write_to_log(self.log_file, "STATE SELECTION RESPONSE\n" + "-"*20)
        self._write_to_log(self.log_file, state_selection_response)
        self.state_history.append(state_selection_response)

        # Extract the selected state ID
        regex = r"Choice:\s*State\s*(\d+)"
        match = re.search(regex, state_selection_response)
        if not match:
            self.state_selection_feedback_msg = "The state choice was malformed. Please provide a valid state choice in the form 'Choice: State <state_id>'."
            return self.current_state
        state_choice = match.group(1)
        selected_node = list(graph.nodes)[int(state_choice)]
        selected_state = graph.nodes[selected_node]["state"]
        self.current_state = selected_state

        # Extract feedback
        if self.current_state != self.next_state:
            # State intervention
            regex = r"Feedback:\s*(.+)"
            match = re.search(regex, state_selection_response)
            if match:
                feedback = match.group(1)
                self.action_feedback_msg += f"\nState Selector Feedback: {feedback}"
    
        # Check if selected state is goal
        model = graph.nodes[selected_node]["model"]
        self.done = model.did_reach_goal(selected_state, goal)

        return selected_state