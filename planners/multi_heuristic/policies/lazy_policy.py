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

class LazyPolicy(PlanPolicy):
    """A plan policy that queries an LLM for a sequence of actions to reach the goal."""
        
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

        # Action plan proposal
        self.action_plan_proposal_prompt_params = kwargs["llm"]["prompts"].get("action_proposal_prompt", {})
        self.action_plan_feedback_msg = ""

        # Planner params
        self.cheap = kwargs["planner"].get("cheap", True)

        self.initial_state = None
        self.final_state = None
        self.action_and_visited_state = [] # (action, visited_state)

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
        self.initial_state = initial_state
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
        # Get action from action proposal response
        action_plan_proposal_prompt = ""
        if self.action_plan_feedback_msg:
            action_plan_proposal_prompt += f"Error Feedback: {self.action_plan_feedback_msg}\n"
            self.action_plan_feedback_msg = ""
        
        # States Visited: ...
        # <action1>: ...
        # <action2>: ...
        # ...
        # <actionN>: ...
        visited_state_description = ""
        for action, visited_state in self.action_and_visited_state:
            visited_state_hash = hash(visited_state)
            if self.state_descriptions.get(visited_state_hash):
                state_description = self.state_descriptions[visited_state_hash]
            else:
                state_str = model.state_to_str(visited_state)
                state_description, _ = self._prompt_llm(state_str, self.state_translation_prompt_params, history=[])
                self.state_descriptions[visited_state_hash] = state_description
            visited_state_description += f"{action}:\n{state_description}\n"
        self.action_and_visited_state = [] # Reset visited states

        # Starting State: ...
        initial_state_hash = hash(self.initial_state)
        if self.state_descriptions.get(initial_state_hash):
            initial_state_description = self.state_descriptions[initial_state_hash]
        else:
            state_str = model.state_to_str(self.initial_state)
            initial_state_description, _ = self._prompt_llm(state_str, self.state_translation_prompt_params, history=[])
            self.state_descriptions[initial_state_hash] = initial_state_description
        
        # Valid Actions: ...
        valid_actions = model.get_valid_actions(state)
        valid_actions_str = "\n".join([f"- {action}" for action in valid_actions])

        # Goal State: ...
        if self.goal_description:
            goal_description = self.goal_description
        else:
            goal_description = model.goal_to_str(state, self.goal)
            goal_description, _ = self._prompt_llm(goal_description, self.state_translation_prompt_params, history=[])
            self.goal_description = goal_description
        
        action_plan_proposal_prompt += f"\nStates Visited:\n{visited_state_description}\n"
        action_plan_proposal_prompt += f"Starting State:\n{initial_state_description}\n"
        action_plan_proposal_prompt += f"Valid Actions:\n{valid_actions_str}\n"
        action_plan_proposal_prompt += f"Goal State:\n{goal_description}\n"

        action_plan_proposal_response, self.chat_history = self._prompt_llm(action_plan_proposal_prompt, self.action_plan_proposal_prompt_params, history=self.chat_history)
        
        # Write to log and append to chat history
        self._write_to_log(self.log_file, "ACTION PLAN PROPOSAL PROMPT\n" + "-"*20)
        self._write_to_log(self.log_file, action_plan_proposal_prompt)
        self.chat_history.append(action_plan_proposal_prompt)
        self._write_to_log(self.log_file, "ACTION PLAN PROPOSAL RESPONSE\n" + "-"*20)
        self._write_to_log(self.log_file, action_plan_proposal_response)

        # Extract action sequence
        action_sequence_regex = r"Action Sequence:\s*(.+)"
        action_sequence_match = re.search(action_sequence_regex, action_plan_proposal_response)
        if not action_sequence_match:
            self.action_plan_feedback_msg = f"The action sequence was malformed. Please provide a valid action sequence in the form 'Action Sequence: <action1>, <action2>, ...'."
            return []
        action_sequence_str = action_sequence_match.group(1)
        action_sequence_list = [action.replace(" ", "") for action in action_sequence_str.split(", ")]

        # Extract reflect
        reflect_regex = r"Reflect:\s*(.+)"
        reflect_match = re.search(reflect_regex, action_plan_proposal_response)
        reflect_str = ""
        if reflect_match:
            reflect_str = f"Reflect: {reflect_match.group(1)}\n"
        think_omitted_str = f"{reflect_str}\nAction Sequence: {action_sequence_str}"
        self.chat_history.append(think_omitted_str)
        self.action_sequence = action_sequence_list
        return action_sequence_list
    
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
        
        curr_state = current_state
        curr_model = model
        for i, action in enumerate(actions):
            # Copy model
            model_copy = deepcopy(curr_model)
            curr_model = model_copy
            # Get valid action
            valid_actions = model_copy.get_valid_actions(curr_state)
            # # Parse action like stack(a:default,b:default) and pick-up(a:default) to stack(a,b) and pick-up(a)
            # typeless_action_regex = r"(\w+)\((.*)\)"
            # typeless_action_match = re.match(typeless_action_regex, action)
            # if typeless_action_match:
            #     action_name = typeless_action_match.group(1)
            #     action_args = typeless_action_match.group(2)
            #     split_args = action_args.split(',')
            #     action_args = [arg.split(':')[0] for arg in split_args]
            #     typeless_action = f"{action_name}({','.join(action_args)})"

            matching_action = list(filter(lambda x: str(x) == action, valid_actions))
            if len(matching_action) == 0:
                valid_actions_str = "\n".join([f"- {action}" for action in valid_actions])
                self.action_plan_feedback_msg = f"The action '{action}' at index {i} was invalid. Below are the actions that were valid at that state:\n{valid_actions_str}"
                return
            matching_action = matching_action[0]
            # Simulate action
            next_state, _, _, _, _ = model_copy.env.step(matching_action)
            graph.add_node(hash(next_state), state=next_state, model=model_copy)
            graph.add_edge(hash(curr_state), hash(next_state), action=action)
            curr_state = next_state
            self.final_state = next_state
            self.action_and_visited_state.append((action, next_state))
        self.done = model_copy.did_reach_goal(next_state, self.goal)
        if not self.done:
            unsatisfied_predicates = []
            for literal in self.goal.literals:
                if literal not in next_state.literals:
                    unsatisfied_predicates.append(str(literal))
            unsatisfied_predicates_str = "\n".join([f"- {predicate}" for predicate in unsatisfied_predicates])
            self.action_plan_feedback_msg = f"The action sequence did not reach the goal. Below are the predicates that were not satisfied:\n{unsatisfied_predicates_str}"

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