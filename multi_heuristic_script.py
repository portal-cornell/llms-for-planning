"""
This script is used to run the multi-heuristic planner on an environment.

Note this script uses Hydra to manage configurations. To create experiments,
look at the `conf/experiments` directory.

To run this script on an example, run the following command in the terminal:
    python multi_heuristic_script.py +experiments=llm_planner
"""
from omegaconf import DictConfig, OmegaConf
import hydra
import random

from prompt_builder.constants import PROMPT_HISTORY_PATH
from prompt_builder.prompt_llm import prompt_llm
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

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_multi_heuristic_planner(cfg: DictConfig) -> None:
    """Run the multi-heuristic planner on the given environment.
    
    Refer to 'conf/experiments` for Hydra configurations corresponding
    to this function.

    Parameters:
        cfg (DictConfig)
            The configuration for the planner.
    
    Side Effects:
        - The planner prints to the console
        - The graph of the plan is saved to a file if specified in the configuration
    """
    populate_messages(cfg)
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs["prompt_fn"] = prompt_llm # Cannot include functions in config

    # Start planning
    plan_policy = NAME_TO_POLICY[cfg.planner.plan_policy](kwargs)
    model = pddlgym_utils.make_pddlgym_model(cfg.planner.env_name)
    model.env.fix_problem_index(cfg.planner.problem_index)
    random.seed(cfg.planner.seed)
    initial_state, _ = model.env.reset()
    goal = initial_state.goal
    action_sequence, graph = plan(plan_policy, model, initial_state, goal, cfg.planner.max_steps)

    if cfg.planner.graph_file is not None:
        visualize_graph(graph, cfg.planner.graph_file)

if __name__ == "__main__":
    run_multi_heuristic_planner()