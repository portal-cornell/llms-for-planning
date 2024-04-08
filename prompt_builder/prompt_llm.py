"""
This module acts as a wrapper for OpenAI's chat API.
"""
import os
import time

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_llms():
    """Returns the available OpenAI LLMs compatible with the Chat API.

    Returns:
        openai_llm_names (List[str])
            The available OpenAI LLMs compatible with the Chat API.
    """
    client = openai.OpenAI()
    openai_models = client.models.list()
    openai_llm_names = [model.id for model in openai_models if 'gpt' in model.id]
    return openai_llm_names

def call_openai_chat(messages, model="gpt-3.5-turbo", temperature=0.0, max_attempts=10, sleep_time=5):
    """Sends chat messages to OpenAI's chat API and returns a response if successful.

    Parameters:
        messages (list)
            A list of dictionaries containing the messages to query the LLM with
        model (str)
            The LLM model to use. Default is "gpt-3.5-turbo"
        temperature (float)
            The LLM temperature to use. Defaults to 0. Note that a temperature of 0 does not 
            guarantee the same response (https://community.openai.com/t/why-the-api-output-is-inconsistent-even-after-the-temperature-is-set-to-0/329541/9).
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

def prompt_llm(user_prompt, messages, model, temperature, **kwargs):
    """Prompt an LLM with the given prompt and return the response.

    Parameters:
        user_prompt (str)
            The user prompt to parse.
        messages (List[Dict[str, str]])
            The messages to query the LLM with.
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
    max_attempts = kwargs.get('max_attempts', 10)
    sleep_time = kwargs.get('sleep_time', 5)
    debug = kwargs.get('debug', False)
    if not debug:
        if model in get_openai_llms():
            messages.append({"role": "user", "content": user_prompt})
            response = call_openai_chat(messages, model, temperature, max_attempts, sleep_time)
        else:
            # TODO(chalo2000): Support open LLM models
            raise NotImplementedError(f"Model {model} is not supported.")
    else:
        response = input("Please input the mocked LLM response: ")
    return response
