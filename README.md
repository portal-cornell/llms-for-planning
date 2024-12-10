# llms-for-planning

## Setup

1. Create and activate your virtual environment. We recommend using [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv). 
   ```sh
   pyenv virtualenv 3.9.13 llms-for-planning
   pyenv activate llms-for-planning
   ```
2. Update the submodules
   ```sh
   git submodule update --init --recursive
   ```
3. Install [pygraphviz](https://pygraphviz.github.io/documentation/stable/install.html) and pip install it
    ```sh
    # Assuming you brew installed graphviz
    export GRAPHVIZ_DIR="$(brew --prefix graphviz)"
    pip install pygraphviz \
        --global-option=build_ext \
        --global-option="-I$GRAPHVIZ_DIR/include" \
        --global-option="-L$GRAPHVIZ_DIR/lib"
    ```
   Add `++planner.graph_file=null` to the command line to disable graph visualization.
4. Install the requirements
   ```sh
   pip install -r requirements.txt
   ```
5. Copy the `.envrctemplate` and fill in the necessary environment variables. Consider using [direnv](https://direnv.net/docs/hook.html).
   ```sh
   cp .envrctemplate .envrc
   # Fill in variables and save file
   direnv allow .
   ```

## Running experiments

This repository uses [Hydra](https://hydra.cc/) for configuration management, [PDDLGym](https://github.com/tomsilver/pddlgym) and [pddlgym_planners](https://github.com/ronuchit/pddlgym_planners) to run PDDL experiments, and [Robotouille](https://github.com/portal-cornell/robotouille) to run Robotouille experiments.

To run an experiment on Blocksworld, use the following command:

```sh
# Run Boomerang on Blocksworld
python main_script.py +experiments=lazy_planner +planner.domain_file=datasets/blocksworld/blocksworld-4ops.pddl +planner.instance_dir=datasets/blocksworld/problems +planner.render_fn_name=blocksworld ++planner.samples=600 ++planner.seed=42 ++planner.max_steps=20 ++llm.temperature=0.7

# Run ReAct on Blocksworld
python main_script.py +experiments=ReAct_planner +planner.domain_file=datasets/blocksworld/blocksworld-4ops.pddl +planner.instance_dir=datasets/blocksworld/problems +planner.render_fn_name=blocksworld ++planner.samples=600 ++planner.seed=42 ++planner.max_steps=20 ++llm.temperature=0.7

# Run ToI-BFS (b=2) on Blocksworld
python main_script.py +experiments=tot_bfs_planner +planner.domain_file=datasets/blocksworld/blocksworld-4ops.pddl +planner.instance_dir=datasets/blocksworld/problems +planner.render_fn_name=blocksworld ++planner.samples=600 ++planner.seed=42 ++planner.max_steps=20  ++planner.candidate_states=2 ++llm.temperature=0.7

# Run ToI-DFS on Blocksworld
python main_script.py +experiments=tot_dfs_planner +planner.domain_file=datasets/blocksworld/blocksworld-4ops.pddl +planner.instance_dir=datasets/blocksworld/problems +planner.render_fn_name=blocksworld ++planner.samples=600 ++planner.seed=42 ++planner.max_steps=20 ++llm.temperature=0.7

# Run classical planners on Blocksworld (all 9 configurations)
python main_script.py --multirun +experiments=fd_planner ++planner.alias="a*-lmcut","wa*-lmcut","bfs-lmcut","a*-ff","wa*-ff","bfs-ff","a*-cg","wa*-cg","bfs-cg" +planner.domain_file=datasets/blocksworld/blocksworld-4ops.pddl +planner.instance_dir=datasets/blocksworld/problems ++planner.samples=600 ++planner.seed=42
```

to run experiments on other PDDL domains, replace `datasets/blocksworld/blocksworld-4ops.pddl` and `datasets/blocksworld/problems` with the appropriate domain and problem files and `+planner.render_fn_name` with the corresponding render function in PDDLGym (if one exists). To debug your setup, decrease the number of steps and samples and add `++llm.model=...` with a cheap model.

To run an experiment on Robotouille, use the following command:

```sh
# Run Boomerang on Robotouille
python main_script.py --multirun +experiments=lazy_planner_robotouille ++planner.env_name=_llms_for_planning/assemble_one_by_one,_llms_for_planning/assemble_parallel,_llms_for_planning/base_cook,_llms_for_planning/base_cut,_llms_for_planning/cheese_burger,_llms_for_planning/cook_patties,_llms_for_planning/cook_then_cut,_llms_for_planning/cut_lettuces,_llms_for_planning/cut_then_cook,_llms_for_planning/lettuce_only_burger ++planner.max_steps=20 ++planner.noisy_randomization=True ++llm.temperature=0.7

# Run ToI-DFS on Robotouille
python main_script.py --multirun +experiments=tot_dfs_planner_robotouille ++planner.env_name=_llms_for_planning/assemble_one_by_one,_llms_for_planning/assemble_parallel,_llms_for_planning/base_cook,_llms_for_planning/base_cut,_llms_for_planning/cheese_burger,_llms_for_planning/cook_patties,_llms_for_planning/cook_then_cut,_llms_for_planning/cut_lettuces,_llms_for_planning/cut_then_cook,_llms_for_planning/lettuce_only_burger ++planner.max_steps=20 ++planner.noisy_randomization=True ++llm.temperature=0.7
```