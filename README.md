# image-stitcher

## Run the GUI

### Pre-setup for linux (e.g. ubuntu)

On linux, ensure you have the necessary system dependencies installed. We
provide a setup script for ubuntu 22.04 LTS that does this automatically:

```
./setup_ubuntu_22_04.sh
```
for other distributions / versions, the exact packages may be slightly
different.

### Pre-setup for mac OS

On mac OS, make sure you have `curl` installed (e.g. `brew install curl`) if
you don't already.

### Run the GUI

Then either double click the `run_gui` script or from this repository's directory run:
```
./run_gui
```

## CLI / developer setup

This repository uses [uv](https://docs.astral.sh/uv/) to manage its environment and dependencies.
If you don't already have it, install from
https://docs.astral.sh/uv/getting-started/installation/ or run the `dev/ensure_uv.sh` script.

## Running via the CLI

You can run registration from the command line via the stitcher_cli module and
its various configuration options. Run:
```
uv run python -m image_stitcher.stitcher_cli --help
```
to see the options and their documentation.

## Devtools

This repository is set up with [ruff](https://docs.astral.sh/ruff/) for linting
and formatting, and [mypy](https://mypy.readthedocs.io) for type checking. The
shell scripts in the dev directory can be used to invoke these tools and should
be run from the repository root.

## Running the GUI manually

```
uv run --extra gui python -m image_stitcher.stitcher_gui
```
