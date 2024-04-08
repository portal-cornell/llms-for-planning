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