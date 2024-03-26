import argparse

from prompt_llm import prompt_llm

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment_name", required=True, help="The name of the experiment for the prompt.")
    argparser.add_argument("--prompt_description", required=True, help="The description of the prompt to test.")
    argparser.add_argument("--prompt_version", required=True, help="The version of the prompt to test.")
    argparser.add_argument("--model", default="gpt-3.5-turbo", help="The LLM model to query.")
    argparser.add_argument("--temperature", default=0.0, type=float, help="The LLM temperature.")
    argparser.add_argument("--max_attempts", default=10, type=int, help="The number of attempts to query the LLM before giving up")
    argparser.add_argument("--debug", action="store_true", help="Whether or not to mock an LLM response")
    argparser.add_argument("--sleep_time", default=5, type=int, help="The number of seconds to sleep after a failed query before requerying")
    args = argparser.parse_args()

    # TODO: Fix relative imports in prompt_llm to allow script usage
    response = prompt_llm(
        args.experiment_name,
        args.prompt_description,
        args.prompt_version,
        args.model,
        args.temperature,
        max_attempts=args.max_attempts,
        sleep_time=args.sleep_time,
        debug=args.debug
    )
    print(response)