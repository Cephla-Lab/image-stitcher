import os
import tempfile
import unittest

from .parameters import OutputFormat, ScanPattern, StitchingParameters


class ParametersTest(unittest.TestCase):
    fixture_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "test_fixtures",
        "parameters_test",
        "parameters.json",
    )

    def test_parsing(self) -> None:
        params = StitchingParameters.from_json_file(self.fixture_file)
        self.assertEqual(params.scan_pattern, ScanPattern.unidirectional)
        self.assertEqual(params.output_format, OutputFormat.ome_zarr)
        self.assertEqual(params.input_folder, "/")

    def test_roundtrip(self) -> None:
        with tempfile.NamedTemporaryFile("w+", delete=True) as f:
            params = StitchingParameters.from_json_file(self.fixture_file)
            params.to_json_file(f.name)
            f.flush()
            f.seek(0)
            contents = f.read()

        with open(self.fixture_file) as f:
            fixture_contents = f.read()

        self.assertEqual(contents, fixture_contents)
