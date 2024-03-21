import argparse
import random
from policies import NAME_TO_POLICY
from v0_single_heuristic import plan, visualize_graph
import pddlgym_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_policy", required=True, choices=NAME_TO_POLICY.keys(), help="The plan policy to use.")
    parser.add_argument("--env_name", required=True, help="The name of the environment.")
    parser.add_argument("--max_steps", type=int, default=20, help="The maximum number of steps to take to reach the goal.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed to use.")
    parser.add_argument("--graph_file", required=False, help="The name of the file to save the graph to.")
    # TODO: Move parser args to config file
    parser.add_argument("--cheap", action="store_true", help="Whether to use the cheap version of the plan policy.")
    parser.add_argument("--num_actions", type=int, default=1, help="The number of actions to propose.")
    args = parser.parse_args()

    kwargs = {"cheap": args.cheap, "num_actions": args.num_actions}
    plan_policy = NAME_TO_POLICY[args.plan_policy](kwargs) # TODO: Move kwargs to config file
    env_name = f"PDDLEnv{args.env_name.capitalize()}-v0"
    model = pddlgym_utils.make_pddlgym_model(env_name)
    random.seed(args.seed)
    initial_state, _ = model.env.reset()
    goal = initial_state.goal
    action_sequence, graph = plan(plan_policy, model, initial_state, goal, max_steps=args.max_steps)

    # Draw graph
    if args.graph_file is not None:
        visualize_graph(graph, args.graph_file)