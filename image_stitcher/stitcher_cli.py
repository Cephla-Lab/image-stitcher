import logging
import sys

from pydantic_settings import CliApp

from image_stitcher.parameters import StitchingParameters
from image_stitcher.stitcher import Stitcher


def main(args: list[str]) -> None:
    params = CliApp.run(StitchingParameters, cli_args=args)
    log_level = logging.DEBUG if params.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    # fsspec debug logs are very spammy
    logging.getLogger("fsspec.local").setLevel(logging.INFO)
    Stitcher(params).run()


if __name__ == "__main__":
    main(sys.argv[1:])
