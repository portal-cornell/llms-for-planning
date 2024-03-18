"""
This module validates and parses the YAML prompt file into messages for an LLM to be prompted with.
"""
import argparse
import yaml

import utils

def prompt_validator_v1(prompt_dict):
    """Validates the prompt YAML.

    Parameters:
        prompt_dict (dict)
            The prompt dictionary from a YAML.

    Raises:
        AssertionError
            If the prompt YAML is invalid.
    """
    assert "version" in prompt_dict, "Prompt YAML must have a version."
    assert "experiment_name" in prompt_dict, "Prompt YAML must have an experiment_name."
    assert "prompt_description" in prompt_dict, "Prompt YAML must have a prompt_description."
    assert "prompt_version" in prompt_dict, "Prompt YAML must have a prompt_version."
    assert "system" in prompt_dict, "Prompt YAML must have a system message."
    assert "instructions" in prompt_dict, "Prompt YAML must have an instruction message."
    if len(prompt_dict.get("examples", [])) > 0:
        observation_count = 0
        response_count = 0
        for example in prompt_dict["examples"]:
            key, _ = list(example.items())[0]
            if key == "observation":
                observation_count += 1
            elif key == "response":
                response_count += 1
        assert observation_count == response_count, "Prompt YAML examples must have an equal number of observations and responses (paired)."

def prompt_parser_v1(prompt_dict):
    """Parses the prompt YAML.

    A v1 prompt YAML includes a system message, an instruction message, and a list of examples.
    For more details, see `prompt_validator_v1`.
    
    Parameters:
        prompt_dict (dict)
            The prompt dictionary from a YAML.

    Returns:
        messages (List[Dict[str, str]])
            The messages to query the LLM with.
    
    Raises:
        AssertionError
            If the prompt YAML is invalid.
    """
    prompt_validator_v1(prompt_dict)
    messages = []
    system_msg = {"role": "system", "content": prompt_dict["system"].strip('\n')}
    messages.append(system_msg)
    instructions_msg = {"role": "user", "content": prompt_dict["instructions"].strip('\n')}
    messages.append(instructions_msg)
    for example in prompt_dict["examples"]:
        key, value = list(example.items())[0]
        if key == "observation":
            message = {"role": "user", "content": value.strip('\n')}
            messages.append(message)
        elif key == "response":
            message = {"role": "assistant", "content": value.strip('\n')}
            messages.append(message)
    return messages

def parse_messages(experiment_name, prompt_description, prompt_version):
    """Parses the messages associated with the provided prompt.

    Parameters:
        experiment_name (str)
            The name of the experiment for the prompt.
        prompt_description (str)
            The description of the prompt to parse.
        prompt_version (str)
            The version of the prompt to parse.
    
    Returns:
        messages (List[Dict[str, str]])
            The messages to query the LLM with.
    
    Raises:
        AssertionError
            If the prompt YAML is invalid.
        NotImplementedError
            - If the prompt parsing version is not supported.
            - If the data tag is not implemented.
    """
    # 1) Get prompt dict
    prompt_path = utils.get_prompt_path(experiment_name, prompt_description, prompt_version)
    with open(prompt_path, "r") as f:
        prompt_dict = yaml.safe_load(f)
    
    # 2) Parse prompt
    major, minor, patch = prompt_version.split(".")
    if major == "1": # e.g. 1.0.0
        messages = prompt_parser_v1(prompt_dict)
    else:
        raise NotImplementedError("Prompt parsing version not supported.")
    
    return messages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True, help="The name of the experiment for the prompt.")
    parser.add_argument("--prompt_description", required=True, help="The description of the prompt to test.")
    parser.add_argument("--prompt_version", required=True, help="The version of the prompt to test.")
    args = parser.parse_args()

    messages = parse_messages(args.experiment_name, args.prompt_description, args.prompt_version)
    print(messages)