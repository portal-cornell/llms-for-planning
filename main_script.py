"""
This script is used to run the multi-heuristic planner on an environment.

Note this script uses Hydra to manage configurations. To create experiments,
look at the `conf/experiments` directory.

To run this script on an example, run the following command in the terminal:
    python main_script.py +experiments=llm_planner
"""
import os
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import hydra
import dill as pickle
import json
import random
import re
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompt_builder.constants import PROMPT_HISTORY_PATH
from prompt_builder.prompt_llm import prompt_llm, get_accumulated_cost
import prompt_builder.serializer as prompt_serializer
import prompt_builder.utils as prompt_utils

from planners.multi_heuristic.v0_single_heuristic import plan, visualize_graph
import planners.multi_heuristic.pddlgym_utils as pddlgym_utils
from planners.multi_heuristic.policies import NAME_TO_POLICY

import logging
logging.getLogger("httpx").setLevel(logging.WARNING) # Suppress LLM HTTP request logging

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

def populate_messages(cfg):
    """Populate the messages for active prompts from the version control directory.
    
    This function finds the planning prompts that aren't using ground truth outputs 
    and populates their messages from the version control directory. Helper prompts
    are populated regardless of the ground truth outputs.

    Parameters:
        cfg (DictConfig)
            The configuration for the planner.
    
    Side Effects:
        - The messages for valid prompts are populated in the configuration
    """
    llm_cfg = cfg.get("llm")
    if llm_cfg is None: return
    for prompt_name in llm_cfg.get("prompts", []):
        prompt = llm_cfg.prompts[prompt_name]
        experiment_name = prompt.get("experiment_name", None)
        prompt_description = prompt.get("prompt_description", None)
        prompt_version = prompt.get("prompt_version", None)
        if experiment_name and prompt_description and prompt_version:
            messages = fetch_messages(experiment_name, prompt_description, prompt_version)
            llm_cfg.prompts[prompt_name]["messages"] = messages

def setup_instance_logging_directory(instance_obj, instance_dir, cfg, kwargs):
    """Creates a logging directory for the instance.
    
    Note that the actual instance directory is passed in along with the instance
    object created for PDDLGym. The filename for the instance_obj is for a file
    that was already deleted (see pddlgym_utils.make_pddlgym_model) so we must 
    retrieve the file from elsewhere.

    Parameters:
        instance (PDDLProblemParser)
            An instance object of a PDDL problem.
        instance_dir (str)
            The directory where actual instances are stored.
        cfg (DictConfig)
            The Hydra configuration for the planner.
        kwargs (Dict[str, Any])
            The configuration dictionary to pass to the planner.
    
    Side Effects:
        - The graph and log file names in the configuration are updated
    """
    instance_abspath = instance_obj.problem_fname
    instance_filename = os.path.basename(instance_abspath)
    instance_name = os.path.splitext(instance_filename)[0]
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    instance_logging_dir = os.path.join(output_dir, instance_name)
    os.makedirs(instance_logging_dir, exist_ok=True)
    if cfg.planner.get("graph_file"):
        graph_filename = os.path.basename(cfg.planner.graph_file)
        kwargs["planner"]["graph_file"] = os.path.join(instance_logging_dir, graph_filename)
    if cfg.planner.get("log_file"):
        log_filename = os.path.basename(cfg.planner.log_file)
        kwargs["planner"]["log_file"] = os.path.join(instance_logging_dir, log_filename)
    actual_instance = os.path.join(instance_dir, instance_filename) # This file exists unlike `instance_abspath`
    instance_abspath_new = os.path.join(instance_logging_dir, instance_filename)
    shutil.copy2(actual_instance, instance_abspath_new)

def parse_log(log_file_path):
    """Parses the log file to get the results of the planner.

    Parameters:
        log_file_path (str)
            The path to the log file to parse.
    
    Returns:
        reached_goal (bool)
            Whether the planner reached the goal.
        optimal_plan (List[str])
            The optimal plan to reach the goal.
        actual_plan (List[str])
            The actual plan taken by the planner.
        total_nodes_expanded (int)
            The total number of nodes expanded by the planner.
        total_edges_expanded (int)
            The total number of edges expanded by the planner.
    """
    with open(log_file_path, "r") as f:
        log = f.read()
    reached_goal = re.search(r"Reached goal: (.*)", log)
    assert reached_goal, f"Reached goal not found in {log_file_path}"
    reached_goal = reached_goal.group(1) == "True"
    optimal_plan = re.search(r"Optimal plan: (.*)", log)
    assert optimal_plan, f"Optimal plan not found in {log_file_path}"
    optimal_plan = json.loads(optimal_plan.group(1))
    actual_plan = re.search(r"Action sequence: (.*)", log)
    assert actual_plan, f"Actual plan not found in {log_file_path}"
    actual_plan = json.loads(actual_plan.group(1))
    total_nodes_expanded = re.search(r"Total nodes expanded: (.*)", log)
    assert total_nodes_expanded, f"Total nodes expanded not found in {log_file_path}"
    total_nodes_expanded = int(total_nodes_expanded.group(1))
    total_edges_expanded = re.search(r"Total edges expanded: (.*)", log)
    assert total_edges_expanded, f"Total edges expanded not found in {log_file_path}"
    total_edges_expanded = int(total_edges_expanded.group(1))
    return reached_goal, optimal_plan, actual_plan, total_nodes_expanded, total_edges_expanded

def create_results_csv(log_name):
    """Creates a CSV file with the results of the planner.

    Parameters:
        log_name (str)
            The name of the log file to search for in the output directory.
    
    Side Effects:
        - A CSV file is created in the output directory
    """
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    csv_filename = os.path.join(output_dir, "results.csv")
    with open(csv_filename, "w") as f:
        f.write("instance,reached_goal,optimal_plan,optimal_length,actual_plan,actual_length,total_nodes_expanded,total_edges_expanded\n")
    
        for dir_path, _, file_names in os.walk(output_dir):
            for file_name in file_names:
                if file_name == log_name:
                    file_path = os.path.join(dir_path, file_name)
                    instance = os.path.basename(dir_path)
                    reached_goal, optimal_plan, actual_plan, total_nodes_expanded, total_edges_expanded = parse_log(file_path)
                    optimal_length = len(optimal_plan)
                    actual_length = len(actual_plan)
                    f.write(f'"{instance}","{reached_goal}","{optimal_plan}","{optimal_length}","{actual_plan}","{actual_length}","{total_nodes_expanded}","{total_edges_expanded}"\n')

def log_planner_results(log_file, optimal_plan, statistics):
    """Logs the results of the classical planner.
    
    Parameters:
        log_file (str)
            The path to the log file to write to.
        optimal_plan (List[str])
            The optimal plan to reach the goal.
        statistics (Dict[str, Any])
            The statistics of the planner.
    
    Side Effects:
        - The results are written to the log file
    """
    if log_file:
        nodes_expanded = statistics.get("num_node_expansions", 0)
        with open(log_file, "a") as f:
            f.write("\n\n")
            f.write(f"Reached goal: {True}\n")
            str_optimal_plan = [str(action) for action in optimal_plan]
            f.write(f"Action sequence: {json.dumps(str_optimal_plan)}\n")
            f.write(f"Total nodes expanded: {nodes_expanded}\n")
            f.write(f"Total edges expanded: {nodes_expanded-1}\n")
            f.write(f"Optimal plan: {json.dumps(str_optimal_plan)}\n")
            f.write("\n")

def planning_loop(i, cfg, model, instance_dir, domain, kwargs):
    instance_obj = model.env.problems[i]
    setup_instance_logging_directory(instance_obj, instance_dir, cfg, kwargs)
    model.env.fix_problem_index(i)
    initial_state, _ = model.env.reset()
    if cfg.planner.plan_policy == "fd":
        log_file = kwargs["planner"].get("log_file")
        optimal_plan, statistics = pddlgym_utils.get_optimal_plan(model.env.domain, initial_state)
        log_planner_results(log_file, optimal_plan, statistics)
        return None, None
    plan_policy = NAME_TO_POLICY[cfg.planner.plan_policy](kwargs)
    goal = initial_state.goal
    reached_goal, action_sequence, graph = plan(plan_policy, model, initial_state, goal, cfg.planner.max_steps, domain)
    
    # Get optimal plan
    optimal_plan, statistics = pddlgym_utils.get_optimal_plan(model.env.domain, initial_state)

    log_file = kwargs["planner"].get("log_file")
    if log_file:
        with open(log_file, "a") as f:
            f.write("\n\n")
            f.write(f"Reached goal: {reached_goal}\n")
            str_action_sequence = [str(action) for action in action_sequence]
            f.write(f"Action sequence: {json.dumps(str_action_sequence)}\n")
            f.write(f"Total nodes expanded: {len(graph.nodes)}\n")
            f.write(f"Total edges expanded: {len(graph.edges)}\n")
            str_optimal_plan = [str(action) for action in optimal_plan]
            f.write(f"Optimal plan: {json.dumps(str_optimal_plan)}\n")
            f.write("\n")
    logging.info(f"Accumulated cost: {get_accumulated_cost()}")
    
    graph_file = kwargs["planner"].get("graph_file")

    if graph_file:
        # TODO: Incorporate into Hydra configuration
        graph_pkl = os.path.splitext(graph_file)[0] + ".pkl"
        with open(graph_pkl, "wb") as f:
            pickle.dump(graph, f)
    return graph, graph_file

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_planner(cfg: DictConfig) -> None:
    """Run a planner on an environment specified in the configuration.
    
    Refer to 'conf/experiments` for Hydra configurations corresponding
    to this function.

    Parameters:
        cfg (DictConfig)
            The configuration for the planner.
    
    Side Effects:
        - The planner prints to the console
        - The graph of the plan is saved to a file if specified in the configuration
    """
    random.seed(cfg.planner.seed)

    # Create planner model
    if cfg.planner.backend == "pddlgym":
        env_name = cfg.planner.get("env_name", None)
        domain_file = cfg.planner.get("domain_file", None)
        instance_dir = cfg.planner.get("instance_dir", None)
        render_fn_name = cfg.planner.get("render_fn_name", None)
        model = pddlgym_utils.make_pddlgym_model(env_name, domain_file, instance_dir, render_fn_name)
    else:
        raise NotImplementedError(f"Backend {cfg.planner.backend} not implemented")
    
    # Prepare kwargs for the plan policy
    populate_messages(cfg)
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs["prompt_fn"] = prompt_llm # Cannot include functions in config

    # Start planning
    num_problems = len(model.env.problems)
    instances = min(cfg.planner.samples, num_problems)
    instance_idx = random.sample(range(num_problems), instances)
    if cfg.planner.get("multi_threaded", False):
        with ThreadPoolExecutor(max_workers=cfg.planner.get("num_workers", 5)) as executor:
            worker = lambda i: planning_loop(i, deepcopy(cfg), deepcopy(model), instance_dir, deepcopy(kwargs))
            futures = [executor.submit(worker, i) for i in instance_idx]
            for future in as_completed(futures):
                graph, graph_file = future.result()
                if graph_file:
                    visualize_graph(graph, graph_file)
    else:
        for i in tqdm(instance_idx):
            graph, graph_file = planning_loop(i, cfg, model, instance_dir, cfg.domain, kwargs)
            if graph_file:
                visualize_graph(graph, graph_file)

    log_name = os.path.basename(cfg.planner.log_file)
    create_results_csv(log_name)

if __name__ == "__main__":
    run_planner()