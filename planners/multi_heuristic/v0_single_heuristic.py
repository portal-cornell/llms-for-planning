"""The V0 single heuristic planner uses an LLM to generate a single plan to reach a goal in a given
environment. Then, starting from an initial state, the plan is used to propose an action(s) to take
in order to reach the goal. Using a model of the environment, the next states are computed and the
plan is used to select the next state to propose actions from. This process is repeated until the
goal is reached.

The V0 single heuristic planner is the simplest planner that we'll use to test the effectiveness
of an LLM as an action proposer and state selector.
"""
import argparse
from copy import deepcopy
import networkx as nx
import random
from tqdm import tqdm

import pddlgym_utils
from policies import NAME_TO_POLICY
from plan_utils import compute_next_states, style_goal_nodes, visualize_graph, plan

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
    parser.add_argument("--index", type=int, default=0, help="The index of the problem to solve.")
    args = parser.parse_args()

    kwargs = {"cheap": args.cheap, "num_actions": args.num_actions}
    plan_policy = NAME_TO_POLICY[args.plan_policy](kwargs) # TODO: Move kwargs to config file
    env_name = f"PDDLEnv{args.env_name.capitalize()}-v0"
    env = pddlgym_utils.make_pddlgym_env(env_name)
    env.fix_problem_index(args.index)
    random.seed(args.seed)
    initial_state, _ = env.reset()
    goal = initial_state.goal
    action_sequence, graph = plan(plan_policy, env, initial_state, goal, max_steps=args.max_steps)

    # Draw graph
    if args.graph_file is not None:
        visualize_graph(graph, args.graph_file)