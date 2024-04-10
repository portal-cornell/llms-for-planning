"""
This module contains utility functions for running the planner scripts.
"""

from prompt_builder.constants import PROMPT_HISTORY_PATH
import prompt_builder.serializer as prompt_serializer
import prompt_builder.utils as prompt_utils

def fetch_messages(experiment_name, prompt_description, prompt_version):
    """Fetches the messages for the prompt from the version control directory.

    Parameters:
        experiment_name (str)
            The name of the experiment for the prompt.
        prompt_description (str)
            The description of the prompt.
        prompt_version (str)
            The version of the prompt.

    Returns:
        messages (List[Dict[str, str]])
            The messages to query the LLM with.
    """
    prompt_path = prompt_utils.get_prompt_path(PROMPT_HISTORY_PATH, experiment_name, prompt_description, prompt_version)
    messages = prompt_serializer.serialize_into_messages(prompt_path)
    return messages