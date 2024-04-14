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
    prompts_to_populate = []
    # Planning prompt population
    if not cfg.llm.ground_truth_plan:
        prompts_to_populate.append(cfg.llm.plan_generation_prompt)
    if not cfg.llm.ground_truth_action:
        prompts_to_populate.append(cfg.llm.action_proposal_prompt)
    if not cfg.llm.ground_truth_state_selection:
        prompts_to_populate.append(cfg.llm.state_translation_prompt)
    # Helper prompt population
    prompts_to_populate.append(cfg.llm.state_translation_prompt)

    for prompt in prompts_to_populate:
        prompt.messages = fetch_messages(prompt.experiment_name, prompt.prompt_description, prompt.prompt_version)

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_multi_heuristic_planner(cfg: DictConfig) -> None:
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