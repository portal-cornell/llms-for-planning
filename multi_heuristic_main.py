import argparse
import random

from prompt_builder.prompt_llm import prompt_llm
from planners.multi_heuristic.v0_single_heuristic import plan, visualize_graph
import planners.multi_heuristic.pddlgym_utils as pddlgym_utils
from planners.multi_heuristic.policies import NAME_TO_POLICY

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument("--experiment_name", required=True, help="The name of the experiment for the prompt.")
    # argparser.add_argument("--prompt_description", required=True, help="The description of the prompt to test.")
    # argparser.add_argument("--prompt_version", required=True, help="The version of the prompt to test.")
    # argparser.add_argument("--model", default="gpt-3.5-turbo", help="The LLM model to query.")
    # argparser.add_argument("--temperature", default=0.0, type=float, help="The LLM temperature.")
    # argparser.add_argument("--max_attempts", default=10, type=int, help="The number of attempts to query the LLM before giving up")
    # argparser.add_argument("--debug", action="store_true", help="Whether or not to mock an LLM response")
    # argparser.add_argument("--sleep_time", default=5, type=int, help="The number of seconds to sleep after a failed query before requerying")
    argparser.add_argument("--plan_policy", required=True, choices=NAME_TO_POLICY.keys(), help="The plan policy to use.")
    argparser.add_argument("--env_name", required=True, help="The name of the environment.")
    argparser.add_argument("--max_steps", type=int, default=20, help="The maximum number of steps to take to reach the goal.")
    argparser.add_argument("--seed", type=int, default=42, help="The random seed to use.")
    argparser.add_argument("--graph_file", required=False, help="The name of the file to save the graph to.")
    argparser.add_argument("--cheap", action="store_true", help="Whether to use the cheap version of the plan policy.")
    argparser.add_argument("--num_actions", type=int, default=1, help="The number of actions to propose.")
    argparser.add_argument("--state_selection", required=True, choices=["pairwise", "total"], help="The state selection method to use.")
    args = argparser.parse_args()

    random.seed(args.seed)
    # TODO(chalo2000): Move all to experiments config file
    expensive_llm = "gpt-4"
    cheap_llm = "gpt-3.5-turbo"
    temperature = 0.7
    max_attempts = 10
    sleep_time = 5

    # user_prompt = ""
    # text_plan = prompt_llm(
    #     user_prompt,
    #     args.experiment_name,
    #     args.prompt_description,
    #     args.prompt_version,
    #     args.model,
    #     args.temperature,
    #     max_attempts=args.max_attempts,
    #     sleep_time=args.sleep_time,
    #     debug=args.debug
    # )
    # print(f"LLM response:\n{text_plan}")

    kwargs = {
        "cheap": args.cheap,
        "expensive_llm": expensive_llm,
        "cheap_llm": cheap_llm,
        "prompt_fn": prompt_llm,
        "plan_generation_prompt": {
            "experiment_name": "plan_generation",
            "prompt_description": "initial",
            "prompt_version": "1.0.0",
            "model": expensive_llm,
            "temperature": temperature,
            "max_attempts": max_attempts,
            "debug": True,
            "sleep_time": sleep_time
        },
        "action_proposal_prompt": {
            "experiment_name": "action_proposal",
            "prompt_description": "initial",
            "prompt_version": "1.0.0",
            "model": expensive_llm,
            "temperature": temperature,
            "max_attempts": max_attempts,
            "debug": False,
            "sleep_time": sleep_time
        },
        "state_selection_prompt": {
            "experiment_name": "state_selection",
            "prompt_description": "initial",
            "prompt_version": "1.0.0" if args.state_selection == "pairwise" else "2.0.0",
            "model": expensive_llm,
            "temperature": temperature,
            "max_attempts": max_attempts,
            "debug": False,
            "sleep_time": sleep_time
        },
        "temperature": temperature,
        "num_actions": args.num_actions
    }
    plan_policy = NAME_TO_POLICY[args.plan_policy](kwargs) # TODO: Move kwargs to config file
    env_name = f"PDDLEnv{args.env_name.capitalize()}-v0"
    model = pddlgym_utils.make_pddlgym_model(env_name)
    random.seed(args.seed)
    initial_state, _ = model.env.reset()
    goal = initial_state.goal
    action_sequence, graph = plan(plan_policy, model, initial_state, goal, args.max_steps)

    if args.graph_file is not None:
        visualize_graph(graph, args.graph_file)