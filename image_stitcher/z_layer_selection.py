"""Z-layer selection utilities for image stitching.

This module provides various strategies for selecting which z-layer(s) to stitch
from a z-stack acquisition.
"""

import logging
from abc import ABC, abstractmethod
from typing import Protocol, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .parameters import MetaKey, AcquisitionMetadata, ZLayerSelection


class ZLayerSelector(ABC):
    """Abstract base class for z-layer selection strategies."""

    @abstractmethod
    def select_z_layers(
        self, metadata: dict['MetaKey', 'AcquisitionMetadata'], num_z: int
    ) -> list[int]:
        """Select which z-layers to include in stitching.

        Args:
            metadata: Dictionary of acquisition metadata keyed by MetaKey
            num_z: Total number of z-layers in the acquisition

        Returns:
            List of z-layer indices to include in stitching
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get a descriptive name for this selection strategy."""
        pass


class MiddleLayerSelector(ZLayerSelector):
    """Selects the middle z-layer from a stack."""

    def select_z_layers(
        self, metadata: dict['MetaKey', 'AcquisitionMetadata'], num_z: int
    ) -> list[int]:
        """Select the middle z-layer.

        Args:
            metadata: Dictionary of acquisition metadata (not used in this implementation)
            num_z: Total number of z-layers

        Returns:
            List containing the index of the middle z-layer
        """
        if num_z <= 0:
            raise ValueError(f"Invalid number of z-layers: {num_z}")

        middle_idx = num_z // 2
        logging.info(
            f"Selected middle z-layer: {middle_idx} (out of {num_z} total layers)"
        )
        return [middle_idx]

    def get_name(self) -> str:
        return "middle_layer"


class AllLayersSelector(ZLayerSelector):
    """Selects all z-layers (default behavior)."""

    def select_z_layers(
        self, metadata: dict['MetaKey', 'AcquisitionMetadata'], num_z: int
    ) -> list[int]:
        """Select all z-layers.

        Args:
            metadata: Dictionary of acquisition metadata (not used in this implementation)
            num_z: Total number of z-layers

        Returns:
            List of all z-layer indices
        """
        return list(range(num_z))

    def get_name(self) -> str:
        return "all_layers"


class SpecificLayerSelector(ZLayerSelector):
    """Selects a specific z-layer by index."""

    def __init__(self, layer_index: int):
        """Initialize with a specific layer index.

        Args:
            layer_index: The z-layer index to select (0-based)
        """
        self.layer_index = layer_index

    def select_z_layers(
        self, metadata: dict['MetaKey', 'AcquisitionMetadata'], num_z: int
    ) -> list[int]:
        """Select a specific z-layer by index.

        Args:
            metadata: Dictionary of acquisition metadata (not used in this implementation)
            num_z: Total number of z-layers

        Returns:
            List containing the specified z-layer index

        Raises:
            ValueError: If the specified index is out of range
        """
        if self.layer_index < 0 or self.layer_index >= num_z:
            raise ValueError(
                f"Z-layer index {self.layer_index} is out of range. "
                f"Valid range is 0 to {num_z - 1}"
            )

        logging.info(
            f"Selected specific z-layer: {self.layer_index} (out of {num_z} total layers)"
        )
        return [self.layer_index]

    def get_name(self) -> str:
        return f"specific_layer_{self.layer_index}"


def filter_metadata_by_z_layers(
    metadata: dict['MetaKey', 'AcquisitionMetadata'], z_layers: list[int]
) -> dict['MetaKey', 'AcquisitionMetadata']:
    """Filter acquisition metadata to include only specified z-layers.

    Args:
        metadata: Original metadata dictionary
        z_layers: List of z-layer indices to include

    Returns:
        Filtered metadata dictionary containing only specified z-layers
    """
    filtered = {}
    for key, value in metadata.items():
        if key.z_level in z_layers:
            filtered[key] = value

    logging.debug(f"Filtered metadata from {len(metadata)} to {len(filtered)} entries")
    return filtered


def create_z_layer_selector(strategy: Union['ZLayerSelection', int, str]) -> ZLayerSelector:
    """Create a z-layer selector based on the specified strategy.

    Args:
        strategy: Name of the selection strategy (ZLayerSelection.ALL, ZLayerSelection.MIDDLE),
                 a numeric index (int), or a string representation ("all", "middle", "0", "1", "2")

    Returns:
        ZLayerSelector instance

    Raises:
        ValueError: If strategy is not recognized
    """
    # Import locally to avoid circular dependency at module load time
    from .parameters import ZLayerSelection

    if isinstance(strategy, ZLayerSelection):
        if strategy == ZLayerSelection.ALL:
            return AllLayersSelector()
        elif strategy == ZLayerSelection.MIDDLE:
            return MiddleLayerSelector()
        # Should not happen if enum is exhaustive
        raise ValueError(f"Unhandled ZLayerSelection enum member: {strategy}")
    elif isinstance(strategy, int):
        return SpecificLayerSelector(strategy)
    elif isinstance(strategy, str):
        # Check if strategy is a numeric string
        if strategy.isdigit():
            layer_index = int(strategy)
            return SpecificLayerSelector(layer_index)

        strategies = {
            "all": AllLayersSelector,
            "middle": MiddleLayerSelector,
        }

        if strategy not in strategies:
            raise ValueError(
                f"Unknown z-layer selection strategy: {strategy}. "
                f"Available strategies: {list(strategies.keys())} or a numeric index, or ZLayerSelection enum"
            )
        return strategies[strategy]()
    else:
        raise TypeError(f"Invalid type for z-layer selection strategy: {type(strategy)}. Expected ZLayerSelection, int, or str.")
