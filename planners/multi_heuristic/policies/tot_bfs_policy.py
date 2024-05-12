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

class ToTBFSPolicy(PlanPolicy):
    """A plan policy that queries an LLM for actions at a state and evaluates states selected through BFS."""
        
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

        # Action proposal
        self.actions_proposal_prompt_params = kwargs["llm"]["prompts"].get("action_proposal_prompt", {})
        self.actions_feedback_msg = ""

        # Value prompt
        self.value_prompt_params = kwargs["llm"]["prompts"].get("state_selection_prompt", {})
        self.value_feedback_msg = ""

        # Planner params
        self.cheap = kwargs["planner"].get("cheap", True)
        self.num_actions = kwargs["planner"].get("num_actions", 1)
        self.max_feedback_steps = kwargs["planner"].get("feedback_steps", 5)
        self.candidate_states = kwargs["planner"].get("candidate_states", 2)
        self.candidates_queue = []

        self.chat_history = []
        self.done = False

        self.initial_state = None
        self.final_state = None
    
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
        self.initial_state = initial_state
        self.candidates_queue.append(initial_state)
        self.goal = goal
        return None
    
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
        i = 0
        feedback_steps = 0
        matching_state_actions = []
        while i < len(self.candidates_queue) and feedback_steps < self.max_feedback_steps:
            candidate_state = self.candidates_queue[i]

            actions_proposal_prompt = ""
            if self.actions_feedback_msg:
                actions_proposal_prompt += f"Error Feedback: {self.actions_feedback_msg}\n"
                self.actions_feedback_msg = ""

            # Valid Actions: ...
            valid_actions = self._actions_to_propose(graph, model, candidate_state)
            valid_actions_str = "\n".join([f"- {action}" for action in valid_actions])
            if len(valid_actions) <= self.num_actions:
                matching_state_actions.append((candidate_state, valid_actions))
                i += 1
                feedback_steps = 0
                self._write_to_log(self.log_file, f"ACTIONS PROPOSAL {i+1} PROMPT\n" + "-"*20)
                self._write_to_log(self.log_file, "[Skip LLM] The number of valid actions is less than the number of actions requested.")
                self._write_to_log(self.log_file, f"Actions:\n{valid_actions_str}\n")
                continue
            
            # Current State: ...
            state_hash = hash(candidate_state)
            if self.state_descriptions.get(state_hash):
                state_description = self.state_descriptions[state_hash]
            else:
                state_str = model.state_to_str(candidate_state)
                state_description, _ = self._prompt_llm(state_str, self.state_translation_prompt_params, history=[])
                self.state_descriptions[state_hash] = state_description

            # Goal State: ...
            if self.goal_description:
                goal_description = self.goal_description
            else:
                goal_description = model.goal_to_str(candidate_state, self.goal)
                goal_description, _ = self._prompt_llm(goal_description, self.state_translation_prompt_params, history=[])
                self.goal_description = goal_description
            
            actions_proposal_prompt += f"Number of Actions: {self.num_actions}\n"
            actions_proposal_prompt += f"Current State:\n{state_description}\n"
            actions_proposal_prompt += f"Valid Actions:\n{valid_actions_str}\n"
            actions_proposal_prompt += f"Goal State:\n{goal_description}\n"

            # no history
            actions_proposal_response, self.chat_history = self._prompt_llm(actions_proposal_prompt, self.actions_proposal_prompt_params, history=[])
            
            # Write to log and append to chat history
            self._write_to_log(self.log_file, f"ACTIONS PROPOSAL {i+1} PROMPT\n" + "-"*20)
            self._write_to_log(self.log_file, actions_proposal_prompt)
            self.chat_history.append(actions_proposal_prompt)
            self._write_to_log(self.log_file, f"ACTIONS PROPOSAL {i+1} RESPONSE\n" + "-"*20)
            self._write_to_log(self.log_file, actions_proposal_response)
            self.chat_history.append(actions_proposal_response)

            # Extract actions
            actions_regex = r"Actions:\s*(.+)"
            actions_match = re.search(actions_regex, actions_proposal_response)
            if not actions_match:
                self.actions_feedback_msg = f"The action sequence was malformed. Please provide a valid action sequence in the form 'Actions: <action1>, <action2>, ...'."
                feedback_steps += 1
                continue
            actions_str = actions_match.group(1)
            actions_list = [action.replace(" ", "") for action in actions_str.split(", ")]
            if len(actions_list) != self.num_actions:
                self.actions_feedback_msg = f"The number of actions provided ({len(actions_list)}) does not match the number of actions requested ({self.num_actions})."
                feedback_steps += 1
                continue
            matching_actions = []
            for action in actions_list:
                matching_action = list(filter(lambda x: str(x) == action, valid_actions))
                if not matching_action:
                    break
                matching_actions.append(matching_action[0])
            if len(matching_actions) != self.num_actions:
                self.actions_feedback_msg = f"Action '{action}' is not a valid action for the current state."
                feedback_steps += 1
                continue
            matching_state_actions.append((candidate_state, matching_actions))
            i += 1
            feedback_steps = 0
        return matching_state_actions
    
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
        
        next_states = []
        for state, actions in actions:
            model = graph.nodes[hash(state)]["model"]
            for action in actions:
                model_copy = deepcopy(model)
                next_state, _, _, _, _ = model_copy.env.step(action)
                graph.add_node(hash(next_state), state=next_state, model=model_copy)
                graph.add_edge(hash(state), hash(next_state), action=action)
                next_states.append(next_state)
        
        # Evaluate next states
        i = 0
        feedback_steps = 0
        rated_states = [] # Ordered list of (rating, order of expansions, state)
        while i < len(next_states) and feedback_steps < self.max_feedback_steps:
            next_state = next_states[i]

            value_prompt = ""
            if self.value_feedback_msg:
                value_prompt += f"Error Feedback: {self.value_feedback_msg}\n"
                self.value_feedback_msg = ""
            
            # Current State: ...
            state_hash = hash(next_state)
            if self.state_descriptions.get(state_hash):
                state_description = self.state_descriptions[state_hash]
            else:
                state_str = model.state_to_str(next_state)
                state_description, _ = self._prompt_llm(state_str, self.state_translation_prompt_params, history=[])
                self.state_descriptions[state_hash] = state_description

            # Goal State: ...
            if self.goal_description:
                goal_description = self.goal_description
            else:
                goal_description = model.goal_to_str(next_state, self.goal)
                goal_description, _ = self._prompt_llm(goal_description, self.state_translation_prompt_params, history=[])
                self.goal_description = goal_description
            
            value_prompt += f"Current State:\n{state_description}\n"
            value_prompt += f"Goal State:\n{goal_description}\n"

            value_response, self.chat_history = self._prompt_llm(value_prompt, self.value_prompt_params, history=[])
            self._write_to_log(self.log_file, f"VALUE PROMPT {i+1}\n" + "-"*20)
            self._write_to_log(self.log_file, value_prompt)
            self.chat_history.append(value_prompt)
            self._write_to_log(self.log_file, f"VALUE RESPONSE {i+1}\n" + "-"*20)
            self._write_to_log(self.log_file, value_response)
            self.chat_history.append(value_response)

            # Extract rating
            rating_regex = r"Rating:\s*(.+)"
            rating_match = re.search(rating_regex, value_response)
            if not rating_match:
                self.value_feedback_msg = f"The rating was malformed. Please provide a valid rating in the form 'Rating: <rating>'."
                feedback_steps += 1
                continue
            rating = rating_match.group(1)
            if rating.lower() == "sure":
                rated_states.append((2, i, next_state))
            elif rating.lower() == "maybe":
                rated_states.append((1, i, next_state))
            elif rating.lower() == "impossible":
                rated_states.append((0, i, next_state))
            else:
                self.value_feedback_msg = f"The rating provided '{rating}' is not valid. Please provide a valid rating that is either 'sure', 'maybe', or 'impossible'."
                feedback_steps += 1
                continue
            i += 1
            feedback_steps = 0
        # Sort by highest rating and lowest order of expansions
        self.candidates_queue = sorted(rated_states, key=lambda x: (x[0], -x[1]), reverse=True)[:self.candidate_states]
        self.candidates_queue = [state for _, _, state in self.candidates_queue]
        for candidate in self.candidates_queue:
            if model.did_reach_goal(candidate, self.goal):
                self.done = True
                self.final_state = candidate
                break

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
        if self.done:
            return self.final_state
        return self.initial_state