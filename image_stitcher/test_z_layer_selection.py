"""Unit tests for z-layer selection functionality."""

import unittest
from unittest.mock import MagicMock

from image_stitcher.parameters import MetaKey, AcquisitionMetadata
from image_stitcher.z_layer_selection import (
    MiddleLayerSelector,
    AllLayersSelector,
    SpecificLayerSelector,
    filter_metadata_by_z_layers,
    create_z_layer_selector,
)


class TestZLayerSelection(unittest.TestCase):
    """Test cases for z-layer selection functionality."""

    def setUp(self):
        """Set up test data."""
        # Create mock metadata for testing
        self.mock_metadata = {}
        for z in range(5):  # 5 z-layers
            for channel in ["405", "488"]:
                key = MetaKey(t=0, region="A1", fov=0, z_level=z, channel=channel)
                meta = MagicMock(spec=AcquisitionMetadata)
                meta.z_level = z
                meta.key = key
                self.mock_metadata[key] = meta

    def test_middle_layer_selector_odd_number(self):
        """Test middle layer selection with odd number of z-layers."""
        selector = MiddleLayerSelector()
        selected = selector.select_z_layers(self.mock_metadata, num_z=5)
        self.assertEqual(selected, [2])  # Middle of 0,1,2,3,4 is 2

    def test_middle_layer_selector_even_number(self):
        """Test middle layer selection with even number of z-layers."""
        selector = MiddleLayerSelector()
        selected = selector.select_z_layers(self.mock_metadata, num_z=4)
        self.assertEqual(selected, [2])  # Middle of 0,1,2,3 is 2 (floor division)

    def test_middle_layer_selector_single_layer(self):
        """Test middle layer selection with single z-layer."""
        selector = MiddleLayerSelector()
        selected = selector.select_z_layers(self.mock_metadata, num_z=1)
        self.assertEqual(selected, [0])

    def test_middle_layer_selector_invalid_input(self):
        """Test middle layer selection with invalid input."""
        selector = MiddleLayerSelector()
        with self.assertRaises(ValueError):
            selector.select_z_layers(self.mock_metadata, num_z=0)
        with self.assertRaises(ValueError):
            selector.select_z_layers(self.mock_metadata, num_z=-1)

    def test_all_layers_selector(self):
        """Test all layers selection."""
        selector = AllLayersSelector()
        selected = selector.select_z_layers(self.mock_metadata, num_z=5)
        self.assertEqual(selected, [0, 1, 2, 3, 4])

    def test_filter_metadata_by_z_layers(self):
        """Test metadata filtering by z-layers."""
        # Filter to only include z-layers 1 and 3
        filtered = filter_metadata_by_z_layers(self.mock_metadata, [1, 3])

        # Check that only z-layers 1 and 3 are in the filtered metadata
        z_levels_in_filtered = set()
        for key in filtered:
            z_levels_in_filtered.add(key.z_level)

        self.assertEqual(z_levels_in_filtered, {1, 3})
        self.assertEqual(len(filtered), 4)  # 2 z-layers × 2 channels

    def test_filter_metadata_empty_selection(self):
        """Test metadata filtering with empty z-layer selection."""
        filtered = filter_metadata_by_z_layers(self.mock_metadata, [])
        self.assertEqual(len(filtered), 0)

    def test_create_z_layer_selector(self):
        """Test factory function for creating selectors."""
        # Test creating valid selectors
        all_selector = create_z_layer_selector("all")
        self.assertIsInstance(all_selector, AllLayersSelector)

        middle_selector = create_z_layer_selector("middle")
        self.assertIsInstance(middle_selector, MiddleLayerSelector)

        # Test invalid strategy
        with self.assertRaises(ValueError):
            create_z_layer_selector("invalid_strategy")

    def test_selector_names(self):
        """Test that selectors return correct names."""
        all_selector = AllLayersSelector()
        self.assertEqual(all_selector.get_name(), "all_layers")

        middle_selector = MiddleLayerSelector()
        self.assertEqual(middle_selector.get_name(), "middle_layer")

    def test_specific_layer_selector(self):
        """Test specific layer selection."""
        # Test selecting layer 0
        selector = SpecificLayerSelector(0)
        selected = selector.select_z_layers(self.mock_metadata, num_z=5)
        self.assertEqual(selected, [0])

        # Test selecting layer 3
        selector = SpecificLayerSelector(3)
        selected = selector.select_z_layers(self.mock_metadata, num_z=5)
        self.assertEqual(selected, [3])

        # Test selecting last layer
        selector = SpecificLayerSelector(4)
        selected = selector.select_z_layers(self.mock_metadata, num_z=5)
        self.assertEqual(selected, [4])

    def test_specific_layer_selector_out_of_range(self):
        """Test specific layer selection with out of range index."""
        # Test negative index
        selector = SpecificLayerSelector(-1)
        with self.assertRaises(ValueError):
            selector.select_z_layers(self.mock_metadata, num_z=5)

        # Test index >= num_z
        selector = SpecificLayerSelector(5)
        with self.assertRaises(ValueError):
            selector.select_z_layers(self.mock_metadata, num_z=5)

    def test_specific_layer_selector_name(self):
        """Test specific layer selector name."""
        selector = SpecificLayerSelector(2)
        self.assertEqual(selector.get_name(), "specific_layer_2")

    def test_create_z_layer_selector_numeric(self):
        """Test factory function with numeric strings."""
        # Test creating selector with numeric string
        selector = create_z_layer_selector("0")
        self.assertIsInstance(selector, SpecificLayerSelector)
        self.assertEqual(selector.layer_index, 0)

        selector = create_z_layer_selector("3")
        self.assertIsInstance(selector, SpecificLayerSelector)
        self.assertEqual(selector.layer_index, 3)


if __name__ == "__main__":
    unittest.main()
