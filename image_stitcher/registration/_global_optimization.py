"""Global optimization module for microscope tile registration.

This module provides functionality for globally optimizing tile positions using a
maximum spanning tree approach. It ensures consistent and optimal positioning of
all tiles while respecting their pairwise relationships.

The module includes:
- Maximum spanning tree computation based on correlation values
- Global position optimization from local translations
- Robust handling of disconnected components
"""
import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from ._typing_utils import Int

# Configure logger
logger = logging.getLogger(__name__)

def validate_grid_dataframe(grid: pd.DataFrame) -> None:
    """Validate grid DataFrame structure and content.
    
    Args:
        grid: DataFrame to validate
        
    Raises:
        ValueError: If DataFrame is invalid or missing required columns
    """
    required_cols = [
        'left_x', 'left_y', 'left_ncc', 'left_valid3',
        'top_x', 'top_y', 'top_ncc', 'top_valid3'
    ]
    
    missing_cols = [col for col in required_cols if col not in grid.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    if grid.empty:
        raise ValueError("Empty grid DataFrame")

def compute_maximum_spanning_tree(grid: pd.DataFrame) -> nx.Graph:
    """Compute maximum spanning tree for global tile position optimization.

    Creates a graph where edges represent tile overlaps, weighted by their
    normalized cross-correlation values. Valid translations get a bonus weight
    to prefer them in the spanning tree.

    Args:
        grid: DataFrame with tile positions and translations

    Returns:
        NetworkX Graph representing the maximum spanning tree

    Raises:
        ValueError: If input data is invalid
    """
    validate_grid_dataframe(grid)
    
    # Create graph
    connection_graph = nx.Graph()
    
    # Add edges with weights
    for i, g in grid.iterrows():
        for direction in ["left", "top"]:
            if not pd.isna(g[direction]):
                # Base weight is the correlation value
                weight = g[f"{direction}_ncc"]
                
                # Add bonus for validated translations
                if g[f"{direction}_valid3"]:
                    weight = weight + 10
                    
                # Add edge with all relevant attributes
                connection_graph.add_edge(
                    i,
                    g[direction],
                    weight=weight,
                    direction=direction,
                    f=i,
                    t=g[direction],
                    y=g[f"{direction}_y"],
                    x=g[f"{direction}_x"],
                )
    
    if not connection_graph.edges:
        raise ValueError("No valid connections between tiles")
        
    # Compute maximum spanning tree
    try:
        tree = nx.maximum_spanning_tree(connection_graph)
        logger.info(f"Created spanning tree with {len(tree.edges)} edges")
        return tree
    except Exception as e:
        raise RuntimeError(f"Failed to compute maximum spanning tree: {e}")

def compute_final_position(
    grid: pd.DataFrame,
    tree: nx.Graph,
    source_index: Int = 0
) -> pd.DataFrame:
    """Compute final tile positions using the maximum spanning tree.

    Traverses the spanning tree to compute consistent global positions for all
    tiles based on their pairwise translations. Handles multiple connected
    components by positioning each relative to its own reference tile.

    Args:
        grid: DataFrame with tile translations
        tree: Maximum spanning tree from compute_maximum_spanning_tree
        source_index: Index of the reference tile (default: 0)

    Returns:
        DataFrame with added x_pos and y_pos columns

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If position computation fails
    """
    if not isinstance(tree, nx.Graph):
        raise ValueError("tree must be a NetworkX Graph")
    if not tree.edges:
        raise ValueError("Empty spanning tree")
        
    # Initialize position columns
    grid['y_pos'] = np.nan
    grid['x_pos'] = np.nan

    # Find connected components
    components = list(nx.connected_components(tree))
    logger.info(f"Found {len(components)} connected components")
    
    try:
        # Process each component
        for component in components:
            # Choose reference tile for this component
            component_source = source_index if source_index in component else min(component)
        
            # Set reference position
            grid.loc[component_source, "y_pos"] = 0
            grid.loc[component_source, "x_pos"] = 0
        
            # Process nodes in breadth-first order
            nodes_to_process = [component_source]
            processed_nodes: Set[int] = set()
            
            while nodes_to_process:
                current_node = nodes_to_process.pop(0)
                processed_nodes.add(current_node)
                
                # Process neighbors
                for neighbor in tree.neighbors(current_node):
                    if neighbor not in processed_nodes:
                        nodes_to_process.append(neighbor)
                        
                        # Get edge properties
                        edge_data = tree.edges[current_node, neighbor]
                        current_pos = (
                            grid.loc[current_node, "y_pos"],
                            grid.loc[current_node, "x_pos"]
                        )
                        
                        # Compute neighbor position
                        if current_node == edge_data["t"]:
                            grid.loc[neighbor, "y_pos"] = current_pos[0] + edge_data["y"]
                            grid.loc[neighbor, "x_pos"] = current_pos[1] + edge_data["x"]
                        else:
                            grid.loc[neighbor, "y_pos"] = current_pos[0] - edge_data["y"]
                            grid.loc[neighbor, "x_pos"] = current_pos[1] - edge_data["x"]

        # Validate and normalize positions
        if grid[["y_pos", "x_pos"]].isna().any().any():
            raise RuntimeError("Failed to compute positions for all tiles")
            
        # Normalize positions to start from (0,0)
        for dim in "yx":
            col = f"{dim}_pos"
            grid[col] = grid[col] - grid[col].min()
            grid[col] = pd.array(np.round(grid[col]), dtype='Int32')

        logger.info("Successfully computed final positions for all tiles")
        return grid
        
    except Exception as e:
        raise RuntimeError(f"Failed to compute final positions: {e}")