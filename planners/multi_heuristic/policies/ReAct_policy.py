"""
This module contains the ReActPolicy class. This LLM policy generates a single action and 
always chooses the next state the action leads to. There is no plan generation in this policy.

The ReAct prompt additionally
- maintains all history of THOUGHTs, ACTs, and OBSs in the prompt
- incorporates environment feedback into the OBS
"""
import os
import openai
import re
import transformers
import torch

from copy import deepcopy
from .policy import PlanPolicy
from .utils import convert_states_to_bitmap_sokoban, map_llm_action_sokoban, map_pddl_to_llm_actions_sokoban

class ReActPolicy(PlanPolicy):
    """A plan policy that queries an LLM to think and act in an environment while receiving feedback."""
    
    FINISH_ACTION = "Finish"

    def __init__(self, kwargs):
        """Initializes the ReAct policy.
        
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
        self.state_translation_prompt_params = kwargs["llm"]["prompts"].get("state_translation_prompt", {})
        if self.state_translation_prompt_params['model'] == "meta-llama/Meta-Llama-3-8B-Instruct":
            self.state_translation_prompt_params['model'] = transformers.pipeline(
                "text-generation",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

        # ReAct prompt
        self.action_proposal_prompt_params = kwargs["llm"]["prompts"].get("action_proposal_prompt", {})
        if self.action_proposal_prompt_params['model'] == "meta-llama/Meta-Llama-3-8B-Instruct":
            self.action_proposal_prompt_params['model'] = transformers.pipeline(
                "text-generation",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        
        self.chat_history = []
        self.truncated_chat_history = [] # Current chat history that fits within the context length
        self.previous_state = None
        self.next_state = None
        
        self.done = False
    
    def is_done(self):
        """Returns whether the policy is done.
        
        The ReAct policy is done when its final action is 'Finish'. This
        
        Returns:
            done (bool)
                Whether the policy is done.
        """
        return self.done
    
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
                    import pdb; pdb.set_trace()
                    assert len(truncated_history) > 2, "The starter user-assistant pair is too long."
                    # Remove one user-assistant pair from the history
                    starter_messages = truncated_history[:2]
                    remaining_messages = truncated_history[4:]
                    truncated_history = starter_messages + remaining_messages
                else:
                    raise e # Raise other errors for user to handle
        return response, truncated_history

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
    
    def _starter_message_template(self, initial_state, goal_state):
        """Returns the starter message template for the ReAct prompt.
        
        Parameters:
            initial_state (object)
                The initial state of the environment.
            goal_state (object)
                The goal state to reach.
        
        Returns:
            starter_message_template (str)
                The starter message template for the ReAct prompt.
        """
        return f"The initial state is the following:\n{initial_state}\n\nThe goal state is the following:\n{goal_state}"
    
    def generate_plan(self, model, initial_state, goal):
        """Generates a plan to reach the goal.

        ReAct does not generate a plan; rather it describes the initial state
        and the goal. This function simply returns the starter message as the
        'plan'.

        Parameters:
            model (Model)
                The model to translate state with.
            initial_state (object)
                The initial state of the environment.
            goal (object)
                The goal to reach.
        
        Returns:
            starter_message (str)
                The starter message for the ReAct prompt.
        
        Side Effects:
            - Prompts the LLM to describe the initial state and goal state.
        """
        if 'typed-sokoban.pddl' in model.env.env._domain_file:
            starter_message = convert_states_to_bitmap_sokoban(initial_state)
        else:
            # Generate initial state description
            initial_state_str = model.state_to_str(initial_state)
            initial_state_description, _ = self._prompt_llm(initial_state_str, self.state_translation_prompt_params)

            # Generate goal state description
            goal_str = model.goal_to_str(goal)
            goal_description, _ = self._prompt_llm(goal_str, self.state_translation_prompt_params)

            # Generate starter message
            starter_message = self._starter_message_template(initial_state_description, goal_description)

        self.next_state = initial_state # Save initial state in case of invalid action at beginning
        return starter_message
    
    def _observation_message_template(self, observation):
        return f"Obs:\n{observation}"
    
    def format_valid_actions(self, model, state):
        if 'typed-sokoban.pddl' in model.env.env._domain_file:
            next_actions = map_pddl_to_llm_actions_sokoban(model.get_valid_actions(state))
        else:
            next_actions = [str(a) for a in model.get_valid_actions(state)]
        
        return next_actions
    
    def propose_actions(self, graph, model, state, plan, provide_actions=False):
        """Proposes an action(s) to take in order to reach the goal.
        
        Parameters:
            graph (nx.DiGraph)
                The graph to propose actions in.
            model (Model)
                The model to propose actions with.
            state (object)
                The current state of the environment.
            plan (object)
                The "plan" to use to propose actions. For ReAct, this is the starter message.
        
        Returns:
            actions (list)
                The proposed actions to take; for ReAct this is a single action.
        """
        user_prompt = ""
        if len(self.chat_history) == 0:
            # Prompt with starter message
            user_prompt = plan
        elif self.previous_state == state:
            # Prompt with no change in state
            user_prompt = self._observation_message_template("No change in state.")
        else:
            # Prompt with observation
            assert hash(state) in graph.nodes, "The current state is not in the graph."
            observation = graph.nodes[hash(state)]["observation"]
            user_prompt = self._observation_message_template(observation)
        
        if provide_actions:
            newline = '\n'
            user_prompt += f"\nYour valid actions are:\n{newline.join(self.format_valid_actions(model, state))}"
        
        self.previous_state = state

        # Get THINK and ACTION from LLM
        thought_and_action, self.truncated_chat_history = self._prompt_llm(user_prompt, self.action_proposal_prompt_params, history=self.truncated_chat_history)
        
        # Update and log user-assistant pair
        self.chat_history.append(user_prompt)
        self.truncated_chat_history.append(user_prompt)
        self._write_to_log(self.log_file, user_prompt)
        self.chat_history.append(thought_and_action)
        self.truncated_chat_history.append(thought_and_action)
        self._write_to_log(self.log_file, thought_and_action)

        # Extract and return ACTION from LLM string response
        regex = r"Action:\s*(.+)"
        match = re.search(regex, thought_and_action)
        if not match:
            self.done = True # Malformed response; kill the planner
            return []
        action = match.group(1)
        action = action.replace(" ", "") # Remove spaces
        valid_actions = model.get_valid_actions(state) + [ReActPolicy.FINISH_ACTION]
        if 'typed-sokoban.pddl' in model.env.env._domain_file:
            action = map_llm_action_sokoban(action)
        matching_action = list(filter(lambda x: str(x) == action, valid_actions))
        return matching_action
    
    def compute_next_states(self, graph, model, current_state, actions):
        """Computes the next states and updates the graph.

        ReAct only ever proposes one action at a time.

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
            - Modifies the graph by adding the next states as nodes and the actions as edges.
            - Saves the next state in self.next_state.
        """
        if len(actions) == 0:
            return # No valid action found
        
        # Simulate action in environment
        action = actions[0] # ReAct only ever proposes one action
        if action == ReActPolicy.FINISH_ACTION:
            self.done = True # ReAct decides when it is done without using model feedback
            return
        model_copy = deepcopy(model)
        next_state, _, _, _, _ = model_copy.env.step(action)
        
        # Format ReAct observation
        next_state_str = model.state_to_str(next_state)
        if 'typed-sokoban.pddl' in model.env.env._domain_file:
            next_state_description = convert_states_to_bitmap_sokoban(next_state)
        else:
            next_state_description, _ = self._prompt_llm(next_state_str, self.state_translation_prompt_params)
        
        # Update graph with next state and action
        graph.add_node(hash(next_state), state=next_state, model=model_copy, observation=next_state_description)
        graph.add_edge(hash(current_state), hash(next_state), action=action)

        # Save the next state
        self.next_state = next_state

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
        """
        # ReAct always chooses the next state it lands in
        return self.next_state