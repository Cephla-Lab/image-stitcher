# image-stitcher

## Setup

This repository uses [uv](https://docs.astral.sh/uv/) to manage its environment and dependencies.
If you don't already have it, install from
https://docs.astral.sh/uv/getting-started/installation/.

## Devtools

This repository is set up with [ruff](https://docs.astral.sh/ruff/) for linting
and formatting, and [mypy](https://mypy.readthedocs.io) for type checking. The
shell scripts in the dev directory can be used to invoke these tools and should
be run from the repository root.

## Running via the CLI

You can run registration from the command line via the stitcher_cli module and
its various configuration options. Run:
```
uv run python -m image_stitcher.stitcher_cli --help
```
to see the options and their documentation.

## Running the GUI

```
uv run --extra gui python -m image_stitcher.stitcher_gui
```