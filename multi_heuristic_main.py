import argparse
import random

from prompt_builder.constants import PROMPT_HISTORY_PATH
from prompt_builder.prompt_llm import prompt_llm
import prompt_builder.serializer as prompt_serializer
import prompt_builder.utils as prompt_utils

from planners.multi_heuristic.v0_single_heuristic import plan, visualize_graph
import planners.multi_heuristic.pddlgym_utils as pddlgym_utils
from planners.multi_heuristic.policies import NAME_TO_POLICY

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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--plan_policy", required=True, choices=NAME_TO_POLICY.keys(), help="The plan policy to use.")
    argparser.add_argument("--env_name", required=True, help="The name of the environment.")
    argparser.add_argument("--problem_index", type=int, default=0, help="The index of the problem to solve (PDDLGym).")
    argparser.add_argument("--max_steps", type=int, default=20, help="The maximum number of steps to take to reach the goal.")
    argparser.add_argument("--seed", type=int, default=42, help="The random seed to use.")
    argparser.add_argument("--graph_file", required=False, help="The name of the file to save the graph to.")
    argparser.add_argument("--log_file", required=False, help="The name of the file to save the log to.")
    argparser.add_argument("--cheap", action="store_true", help="Whether to use the cheap version of the plan policy.")
    argparser.add_argument("--num_actions", type=int, default=1, help="The number of actions to propose.")
    args = argparser.parse_args()

    random.seed(args.seed)
    # TODO(chalo2000): Move all to experiments config file
    expensive_llm = "gpt-4"
    cheap_llm = "gpt-3.5-turbo"
    temperature = 0.7
    max_attempts = 10
    sleep_time = 5
    kwargs = {
        "cheap": args.cheap,
        "expensive_llm": expensive_llm,
        "cheap_llm": cheap_llm,
        "prompt_fn": prompt_llm,
        "ground_truth_plan": True,
        "ground_truth_action": False,
        "ground_truth_state_selection": True,
        "state_translation_prompt": {
            "messages": fetch_messages("state_translation_blocksworld", "initial", "1.0.0"),
            "model": expensive_llm,
            "temperature": temperature,
            "max_attempts": max_attempts,
            "debug": False,
            "sleep_time": sleep_time 
        },
        "action_proposal_prompt": {
            "messages": fetch_messages("action_proposal", "initial", "1.0.0"),
            "model": expensive_llm,
            "temperature": temperature,
            "max_attempts": max_attempts,
            "debug": False,
            "sleep_time": sleep_time
        },
        "temperature": temperature,
        "num_actions": args.num_actions,
        "log_file": args.log_file
    }
    plan_policy = NAME_TO_POLICY[args.plan_policy](kwargs) # TODO: Move kwargs to config file
    env_name = f"PDDLEnv{args.env_name.capitalize()}-v0"
    model = pddlgym_utils.make_pddlgym_model(env_name)
    model.env.fix_problem_index(args.problem_index)
    random.seed(args.seed)
    initial_state, _ = model.env.reset()
    goal = initial_state.goal
    action_sequence, graph = plan(plan_policy, model, initial_state, goal, args.max_steps)

    if args.graph_file is not None:
        visualize_graph(graph, args.graph_file)