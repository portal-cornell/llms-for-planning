"""
This module acts as a wrapper for OpenAI's chat API.
"""
import argparse
import os
import time

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from .constants import PROMPT_HISTORY_PATH
from . import serializer
from . import utils

def call_openai_chat(messages, model="gpt-3.5-turbo", temperature=0.0, max_attempts=10, sleep_time=5):
    """Sends chat messages to OpenAI's chat API and returns a response if successful.

    Parameters:
        messages (list)
            A list of dictionaries containing the messages to query the LLM with
        model (str)
            The LLM model to use. Default is "gpt-3.5-turbo"
        temperature (float)
            The LLM temperature to use. Defaults to 0. Note that a temperature of 0 does not 
            guarantee the same response (https://platform.openai.com/docs/models/gpt-3-5).
        max_attempts (int)
            The number of attempts to query the LLM before giving up
        sleep_time (int)
            The number of seconds to sleep after a failed query before requerying

    Returns:
        response (Optional[dict])
            The response from OpenAI's chat API, if any.
    
    Raises:
        AssertionError
            If the OpenAI API key is invalid
    """
    client = openai.OpenAI()

    num_attempts = 0
    response = None
    while response is None and num_attempts < max_attempts:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            response = response.choices[0].message.content
        except openai.OpenAIError as e:
            assert not isinstance(e, openai.AuthenticationError), "Invalid OpenAI API key"
            print(e)
            print(f"API Error #{num_attempts+1}: Sleeping...")
            time.sleep(sleep_time)
            num_attempts += 1
    return response

def prompt_llm(user_prompt, experiment_name, prompt_description, prompt_version, model, temperature, **kwargs):
    """Prompt an LLM with the given prompt and return the response.

    Parameters:
        user_prompt (str)
            The user prompt to parse.
        experiment_name (str)
            The name of the experiment for the prompt.
        prompt_description (str)
            The description of the prompt to parse.
        prompt_version (str)
            The version of the prompt to parse.
        model (str)
            The LLM model to use.
        temperature (float)
            The LLM temperature to use.
        kwargs (Dict[str, any])
            Optional parameters for LLM querying such as:
                max_attempts (int)
                    The number of attempts to query the LLM before giving up
                sleep_time (int)
                    The number of seconds to sleep after a failed query before requerying
                debug (bool)
                    Whether or not to mock an LLM response

    Returns:
        response (Optional[dict])
            The response from OpenAI's chat API, if any.
    
    Raises:
        AssertionError
            If the prompt YAML is invalid.
        NotImplementedError
            - If the prompt parsing version is not supported.
            - If the data tag is not implemented.
    """
    prompt_path = utils.get_prompt_path(PROMPT_HISTORY_PATH, experiment_name, prompt_description, prompt_version)
    messages = serializer.parse_messages(prompt_path)
    messages.append({"role": "user", "content": user_prompt})
    max_attempts = kwargs.get('max_attempts', 10)
    sleep_time = kwargs.get('sleep_time', 5)
    debug = kwargs.get('debug', False)
    if not debug:
        # TODO: Support OpenLLM
        # if model in openai.models?
        response = call_openai_chat(messages, model, temperature, max_attempts, sleep_time)
    else:
        response = input("Please input the mocked LLM response: ")
    return response
