# image-stitcher
## Setup
### Pre-setup for linux (e.g. ubuntu)

On linux, ensure you have the necessary system dependencies installed and
conda environment setup. We provide a setup script for ubuntu 22.04 LTS
that does this automatically:

```
./setup_ubuntu_22_04.sh
```

This will set up a conda environment called `image-stitcher` with all the required dependencies,
and should let you run the examples below when activated with `conda activate image-stitcher`.

### Pre-setup for Non-Ubuntu OS

For other environments, you will need to manually replicate the setup steps in `./setup_ubuntu_22_04.sh`.

We use [BaSiCPy](https://basicpy.readthedocs.io/en/latest/installation.html) which can be finicky.  Specifically
with respect to its jax depencency.  So far what has worked best is making sure to follow the exact suggestions
in their installation page with respect to installing jax separately, or just running `pip install basicpy`.

## Running The Stitcher
### Run the GUI

Activate the `image-stitcher` conda environment, then run `run_gui` script:
```
./run_gui
```

To run the gui manually without the helper script, you can run the following from this directory:
```
python -m image_stitcher.stitcher_gui
```

## Running via the CLI

You can run registration from the command line via the stitcher_cli module and
its various configuration options. Run (with the conda `image-stitcher` environment activated):
```
python -m image_stitcher.stitcher_cli --help
```
to see the options and their documentation.

## Devtools

This repository is set up with [ruff](https://docs.astral.sh/ruff/) for linting
and formatting, and [mypy](https://mypy.readthedocs.io) for type checking. The
shell scripts in the dev directory can be used to invoke these tools and should
be run from the repository root.
