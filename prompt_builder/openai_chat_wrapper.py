"""
This module acts as a wrapper for OpenAI's chat API.
"""
import os
import time

import openai
import json
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_openai_chat(messages, model="gpt-3.5-turbo", temperature=0.0):
    """
    Sends a request with a chat conversation to OpenAI's chat API and returns a response.

    Parameters:
        messages (list)
            A list of dictionaries containing the messages to send to the chatbot.
        model (str)
            The model to use for the chatbot. Default is "gpt-3.5-turbo".
        temperature (float)
            The temperature to use for the chatbot. Defaults to 0. Note that a temperature
            of 0 does not guarantee the same response (https://platform.openai.com/docs/models/gpt-3-5).

    Returns:
        response (Optional[dict])
            The response from OpenAI's chat API, if any.
    """
    client = openai.OpenAI()

    num_attempts = 0

    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response
        except openai.AuthenticationError as e:
            print(e)
            return None
        except openai.RateLimitError as e:
            print(e)
            print("Sleeping for 5 seconds...")
            time.sleep(5)
            num_attempts += 1
        except Exception as e:
            print(e)
            print("Sleeping for 2 seconds...")
            time.sleep(2)
            num_attempts += 1
