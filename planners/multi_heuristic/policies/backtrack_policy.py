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

class BacktrackPolicy(PlanPolicy):
    """A plan policy that queries an LLM to propose states and actions"""
        
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
        self.goal_description = "" # Cache for goal description

        # State translation
        self.state_translation_prompt_params = kwargs["llm"]["prompts"].get("state_translation_prompt", {})

        # State-action proposal
        self.state_action_proposal_prompt_params = kwargs["llm"]["prompts"].get("action_proposal_prompt", {})
        self.state_action_feedback_msg = ""

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
    
    def _prompt_llm(self, user_prompt, params, history):
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
        self.current_state = initial_state
        return goal
    
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
        assert False, f"State {state} not found in graph"
    
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
        
        # Get action from action proposal response
        state_action_proposal_prompt = ""
        if self.state_action_feedback_msg:
            state_action_proposal_prompt += f"Error Feedback: {self.state_action_feedback_msg}\n"
            self.state_action_feedback_msg = ""
        
        # State i
        state_id = self._get_state_id(graph, state)
        did_visit = "(visited previously)" if self.state_descriptions.get(hash(state)) else ""
        state_action_proposal_prompt += f"State {state_id} {did_visit}\n"


        # State Description: ...
        if self.state_descriptions.get(hash(state)):
            state_description = self.state_descriptions[hash(state)]
        else:
            state_str = model.state_to_str(state)
            state_description, _ = self._prompt_llm(state_str, self.state_translation_prompt_params, history=[])
            self.state_descriptions[hash(state)] = state_description

        # Goal: ...
        # if self.goal_description:
        #     goal_description = self.goal_description
        # else:
        #     goal_description = model.goal_to_str(state, plan) # Plan is the goal
        #     goal_description, _ = self._prompt_llm(goal_description, self.state_translation_prompt_params)
        #     self.goal_description = goal_description
        goal_description = model.goal_to_str(state, plan) # Plan is the goal

        # Valid Actions: ...
        valid_actions = model.get_valid_actions(state)
        valid_actions_str = "\n".join([f"- {action}" for action in valid_actions])

        state_action_proposal_prompt += f"State Description:\n{state_description}\n"
        state_action_proposal_prompt += f"Goal:\n{goal_description}\n"
        state_action_proposal_prompt += f"Valid Actions:\n{valid_actions_str}\n"

        state_action_proposal_response, self.chat_history = self._prompt_llm(state_action_proposal_prompt, self.state_action_proposal_prompt_params, history=self.chat_history)
        
        # Write to log and append to chat history
        self._write_to_log(self.log_file, "STATE-ACTION PROPOSAL PROMPT\n" + "-"*20)
        self._write_to_log(self.log_file, state_action_proposal_prompt)
        self.chat_history.append(state_action_proposal_prompt)
        self._write_to_log(self.log_file, "STATE-ACTION PROPOSAL RESPONSE\n" + "-"*20)
        self._write_to_log(self.log_file, state_action_proposal_response)
        # self.chat_history.append(state_action_proposal_response)
        
        # Extract state and action regexes
        state_regex = r"State:\s*State\s*(\d+)"
        state_match = re.search(state_regex, state_action_proposal_response)
        if not state_match:
            self.state_action_feedback_msg = f"The state was malformed. Please provide a valid state in the form 'State: State <state_id>'."
            self.chat_history.append(f"Error Feedback: {self.state_action_feedback_msg}\n")
            return []
        
        action_regex = r"Action:\s*(.+)"
        action_match = re.search(action_regex, state_action_proposal_response)
        if not action_match:
            self.state_action_feedback_msg = "The action was malformed. Please provide a valid action in the form 'Action: <action>'."
            self.chat_history.append(f"Error Feedback: {self.state_action_feedback_msg}\n")
            return []
        self.chat_history.append(f"State: State {state_match.group(1)}\nAction: {action_match.group(1)}\n")
        
        # Extract actual state and action
        proposed_state_id = state_match.group(1)
        if not proposed_state_id.isdigit() or int(proposed_state_id) >= len(graph.nodes):
            self.state_action_feedback_msg = f"The state ID '{proposed_state_id}' was invalid. Please provide a valid state ID."
            return []
        proposed_node = list(graph.nodes)[int(proposed_state_id)]
        proposed_state = graph.nodes[proposed_node]["state"]

        proposed_action = action_match.group(1)
        stripped_action = proposed_action.replace(" ", "") # Remove spaces (if any)
        valid_actions = model.get_valid_actions(state)
        matching_action = list(filter(lambda x: str(x) == stripped_action, valid_actions))
        if len(matching_action) == 0:
            self.state_action_feedback_msg = f"The action '{proposed_action}' was invalid. Please provide a valid action from the list."
            return []
        
        state_action_dict = {
            "state": proposed_state,
            "actions": matching_action
        }

        return state_action_dict
    
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
            self.current_state = next_state
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
        return self.current_state