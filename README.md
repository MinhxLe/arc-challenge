# arc

## Set up
1. `create a virtualenv.`
2. `pip install -r requirements.txt && pip install -e .`
3. Outside of this project, `git clone https://github.com/michaelhodel/arc-dsl.git`
4. Add this line to your shell config: `export PYTHONPATH="[path to arc-dsl folder]:$PYTHONPATH"`. Replace `[path to arc-dsl folder]` with yours, e.g. `/Users/your/Projects/arc-dsl`. [To-do: figure out how to deconflict namespace with our own repo - maybe submodule arc-dsl?]