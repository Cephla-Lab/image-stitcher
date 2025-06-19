# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle, FancyArrowPatch
# import matplotlib.patches as mpatches
# import pandas as pd
# from pathlib import Path
# import os
# import re
# import sys
# from typing import Union, Dict, List, Tuple

# # Constants for file handling
# DEFAULT_FOV_RE = re.compile(r"(?P<region>\w+)_(?P<fov>0|[1-9]\d*)_(?P<z_level>\d+)_", re.I)

# def extract_tile_indices(filenames: List[str], coords_df: pd.DataFrame) -> Tuple[List[int], List[int], Dict[str, int]]:
#     """Extract row and column indices for each tile from filenames.
    
#     Parameters
#     ----------
#     filenames : List[str]
#         List of filenames
#     coords_df : pd.DataFrame
#         DataFrame containing coordinates
        
#     Returns
#     -------
#     Tuple[List[int], List[int], Dict[str, int]]
#         Lists of row and column indices, and mapping from filename to index
#     """
#     rows = []
#     cols = []
#     filename_to_index = {}
    
#     for i, filename in enumerate(filenames):
#         match = DEFAULT_FOV_RE.search(filename)
#         if match:
#             fov = int(match.group('fov'))
#             # Find the corresponding row in coords_df
#             coord_idx = coords_df[coords_df['fov'] == fov].index[0]
#             rows.append(coord_idx)
#             cols.append(i)  # Use the index as column number
#             filename_to_index[filename] = i
    
#     return rows, cols, filename_to_index

# def create_registration_scatterplots(
#     timepoint_dir: Path,
#     original_coords_dir: Path,
#     output_dir: Path,
#     region: str
# ) -> None:
#     """Create scatterplots comparing original vs registered coordinates for a region.
    
#     Parameters
#     ----------
#     timepoint_dir : Path
#         Directory containing the timepoint data (e.g., "0", "1", etc.)
#     original_coords_dir : Path
#         Directory containing original coordinates backups
#     output_dir : Path
#         Directory to save scatterplots
#     region : str
#         Region name to process
#     """
#     # Read original coordinates
#     timepoint = timepoint_dir.name
#     original_coords_path = original_coords_dir / f"original_coordinates_{timepoint}.csv"
#     if not original_coords_path.exists():
#         raise FileNotFoundError(f"Original coordinates not found at {original_coords_path}")
    
#     original_df = pd.read_csv(original_coords_path)
#     original_df = original_df[original_df['region'] == region]
    
#     # Read registered coordinates
#     registered_coords_path = timepoint_dir / "coordinates.csv"
#     if not registered_coords_path.exists():
#         raise FileNotFoundError(f"Registered coordinates not found at {registered_coords_path}")
    
#     registered_df = pd.read_csv(registered_coords_path)
#     registered_df = registered_df[registered_df['region'] == region]
    
#     # Get list of image files for this region using DEFAULT_FOV_RE
#     image_files = []
#     for file in timepoint_dir.iterdir():
#         if file.is_file():
#             match = DEFAULT_FOV_RE.search(file.name)
#             if match and match.group('region') == region:
#                 image_files.append(file)
    
#     if not image_files:
#         raise FileNotFoundError(f"No image files found for region {region} in {timepoint_dir}")
    
#     # Create a mapping from FOV to filename
#     fov_to_filename = {}
#     for file in image_files:
#         match = DEFAULT_FOV_RE.search(file.name)
#         if match:
#             fov = int(match.group('fov'))
#             fov_to_filename[fov] = file.name
    
#     # Filter DataFrames to only include FOVs that have corresponding image files
#     original_df = original_df[original_df['fov'].isin(fov_to_filename.keys())]
#     registered_df = registered_df[registered_df['fov'].isin(fov_to_filename.keys())]
    
#     # Add filename column to DataFrames
#     original_df['filename'] = original_df['fov'].map(fov_to_filename)
#     registered_df['filename'] = registered_df['fov'].map(fov_to_filename)
    
#     # Sort DataFrames by FOV to ensure consistent ordering
#     original_df = original_df.sort_values('fov')
#     registered_df = registered_df.sort_values('fov')
    
#     # Extract coordinates and convert to microns
#     original_x = original_df['x (mm)'].values * 1000  # Convert to microns
#     original_y = original_df['y (mm)'].values * 1000  # Convert to microns
#     registered_x = registered_df['x (mm)'].values * 1000  # Convert to microns
#     registered_y = registered_df['y (mm)'].values * 1000  # Convert to microns
    
#     # Determine grid size from unique x and y coordinates
#     unique_x = np.unique(original_x)
#     unique_y = np.unique(original_y)
#     n_cols = len(unique_x)
#     n_rows = len(unique_y)
    
#     # Create a mapping from (x,y) to grid position
#     x_to_col = {x: i for i, x in enumerate(sorted(unique_x))}
#     y_to_row = {y: i for i, y in enumerate(sorted(unique_y))}
    
#     # Create sparse grids filled with NaN
#     orig_x_grid = np.full((n_rows, n_cols), np.nan)
#     orig_y_grid = np.full((n_rows, n_cols), np.nan)
#     reg_x_grid = np.full((n_rows, n_cols), np.nan)
#     reg_y_grid = np.full((n_rows, n_cols), np.nan)
    
#     # Fill in the grids with actual coordinates
#     for x, y, reg_x, reg_y in zip(original_x, original_y, registered_x, registered_y):
#         row = y_to_row[y]
#         col = x_to_col[x]
#         orig_x_grid[row, col] = x
#         orig_y_grid[row, col] = y
#         reg_x_grid[row, col] = reg_x
#         reg_y_grid[row, col] = reg_y
    
#     # Calculate displacements for each tile
#     dx_all = reg_x_grid - orig_x_grid
#     dy_all = reg_y_grid - orig_y_grid
    
#     # Calculate dx, dy for horizontally adjacent tiles
#     horizontal_dx = []
#     horizontal_dy = []
#     horizontal_positions = []
    
#     for row in range(n_rows):
#         for col in range(n_cols - 1):
#             # Skip if either tile is missing
#             if np.isnan(orig_x_grid[row, col]) or np.isnan(orig_x_grid[row, col + 1]):
#                 continue
                
#             # Get positions for current tile and its right neighbor
#             x1_orig = orig_x_grid[row, col]
#             y1_orig = orig_y_grid[row, col]
#             x2_orig = orig_x_grid[row, col + 1]
#             y2_orig = orig_y_grid[row, col + 1]
            
#             x1_reg = reg_x_grid[row, col]
#             y1_reg = reg_y_grid[row, col]
#             x2_reg = reg_x_grid[row, col + 1]
#             y2_reg = reg_y_grid[row, col + 1]
            
#             # Expected spacing in original grid
#             expected_dx = x2_orig - x1_orig
#             expected_dy = y2_orig - y1_orig
            
#             # Actual spacing after registration
#             actual_dx = x2_reg - x1_reg
#             actual_dy = y2_reg - y1_reg
            
#             # Deviation from expected spacing
#             dx = actual_dx - expected_dx
#             dy = actual_dy - expected_dy
            
#             horizontal_dx.append(dx)
#             horizontal_dy.append(dy)
#             horizontal_positions.append((row, col))
    
#     # Calculate dx, dy for vertically adjacent tiles
#     vertical_dx = []
#     vertical_dy = []
#     vertical_positions = []
    
#     for row in range(n_rows - 1):
#         for col in range(n_cols):
#             # Skip if either tile is missing
#             if np.isnan(orig_x_grid[row, col]) or np.isnan(orig_x_grid[row + 1, col]):
#                 continue
                
#             # Get positions for current tile and its bottom neighbor
#             x1_orig = orig_x_grid[row, col]
#             y1_orig = orig_y_grid[row, col]
#             x2_orig = orig_x_grid[row + 1, col]
#             y2_orig = orig_y_grid[row + 1, col]
            
#             x1_reg = reg_x_grid[row, col]
#             y1_reg = reg_y_grid[row, col]
#             x2_reg = reg_x_grid[row + 1, col]
#             y2_reg = reg_y_grid[row + 1, col]
            
#             # Expected spacing in original grid
#             expected_dx = x2_orig - x1_orig
#             expected_dy = y2_orig - y1_orig
            
#             # Actual spacing after registration
#             actual_dx = x2_reg - x1_reg
#             actual_dy = y2_reg - y1_reg
            
#             # Deviation from expected spacing
#             dx = actual_dx - expected_dx
#             dy = actual_dy - expected_dy
            
#             vertical_dx.append(dx)
#             vertical_dy.append(dy)
#             vertical_positions.append((row, col))
    
#     # Create the main scatter plot figure
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     fig.suptitle(f'Adjacent Tile Registration Displacements (dx, dy)\nTimepoint {timepoint}, Region {region}', fontsize=16)
    
#     # 1. Horizontal adjacent tiles scatter plot
#     ax = axes[0]
#     scatter_h = ax.scatter(horizontal_dx, horizontal_dy, alpha=0.7, s=100, c='blue', edgecolors='black', label='Horizontal pairs')
#     ax.set_xlabel('dx (μm)')
#     ax.set_ylabel('dy (μm)')
#     ax.set_title(f'Horizontal Adjacent Tiles\n({len(horizontal_dx)} pairs)')
#     ax.grid(True, alpha=0.3)
#     ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
#     ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
#     # Add statistics
#     mean_h_dx = np.mean(horizontal_dx)
#     mean_h_dy = np.mean(horizontal_dy)
#     std_h_dx = np.std(horizontal_dx)
#     std_h_dy = np.std(horizontal_dy)
#     ax.text(0.05, 0.95, f'Mean dx: {mean_h_dx:.2f} μm\nStd dx: {std_h_dx:.2f} μm\n'
#                         f'Mean dy: {mean_h_dy:.2f} μm\nStd dy: {std_h_dy:.2f} μm',
#             transform=ax.transAxes, verticalalignment='top',
#             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
#     # 2. Vertical adjacent tiles scatter plot
#     ax = axes[1]
#     scatter_v = ax.scatter(vertical_dx, vertical_dy, alpha=0.7, s=100, c='red', edgecolors='black', label='Vertical pairs')
#     ax.set_xlabel('dx (μm)')
#     ax.set_ylabel('dy (μm)')
#     ax.set_title(f'Vertical Adjacent Tiles\n({len(vertical_dx)} pairs)')
#     ax.grid(True, alpha=0.3)
#     ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
#     ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
#     # Add statistics
#     mean_v_dx = np.mean(vertical_dx)
#     mean_v_dy = np.mean(vertical_dy)
#     std_v_dx = np.std(vertical_dx)
#     std_v_dy = np.std(vertical_dy)
#     ax.text(0.05, 0.95, f'Mean dx: {mean_v_dx:.2f} μm\nStd dx: {std_v_dx:.2f} μm\n'
#                         f'Mean dy: {mean_v_dy:.2f} μm\nStd dy: {std_v_dy:.2f} μm',
#             transform=ax.transAxes, verticalalignment='top',
#             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
#     # 3. Combined scatter plot
#     ax = axes[2]
#     ax.scatter(horizontal_dx, horizontal_dy, alpha=0.7, s=100, c='blue', edgecolors='black', label='Horizontal pairs')
#     ax.scatter(vertical_dx, vertical_dy, alpha=0.7, s=100, c='red', edgecolors='black', label='Vertical pairs')
#     ax.set_xlabel('dx (μm)')
#     ax.set_ylabel('dy (μm)')
#     ax.set_title(f'All Adjacent Tiles\n({len(horizontal_dx) + len(vertical_dx)} pairs total)')
#     ax.grid(True, alpha=0.3)
#     ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
#     ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
#     ax.legend()
    
#     # Add combined statistics
#     all_dx = horizontal_dx + vertical_dx
#     all_dy = horizontal_dy + vertical_dy
#     mean_all_dx = np.mean(all_dx)
#     mean_all_dy = np.mean(all_dy)
#     std_all_dx = np.std(all_dx)
#     std_all_dy = np.std(all_dy)
#     ax.text(0.05, 0.95, f'Mean dx: {mean_all_dx:.2f} μm\nStd dx: {std_all_dx:.2f} μm\n'
#                         f'Mean dy: {mean_all_dy:.2f} μm\nStd dy: {std_all_dy:.2f} μm',
#             transform=ax.transAxes, verticalalignment='top',
#             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
#     plt.tight_layout()
    
#     # Save the plot
#     output_path = output_dir / f"registration_scatterplot_tp{timepoint}_region{region}.png"
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()

# def generate_all_registration_scatterplots(base_dir: Union[str, Path]) -> None:
#     """Generate scatterplots for all timepoints and regions.
    
#     Parameters
#     ----------
#     base_dir : Union[str, Path]
#         Base directory containing timepoint subdirectories
#     """
#     base_dir = Path(base_dir)
#     original_coords_dir = base_dir / "original_coordinates"
#     scatterplots_dir = base_dir / "scatterplots"
    
#     # Create scatterplots directory
#     scatterplots_dir.mkdir(exist_ok=True)
    
#     # Find all timepoint directories
#     timepoint_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
#     for tp_dir in timepoint_dirs:
#         timepoint = tp_dir.name
#         print(f"\nProcessing timepoint {timepoint}")
        
#         # Create timepoint-specific scatterplot directory
#         tp_scatterplots_dir = scatterplots_dir / timepoint
#         tp_scatterplots_dir.mkdir(exist_ok=True)
        
#         # Read coordinates to get regions
#         coords_path = tp_dir / "coordinates.csv"
#         if not coords_path.exists():
#             print(f"Warning: No coordinates.csv found in timepoint {timepoint}, skipping")
#             continue
            
#         coords_df = pd.read_csv(coords_path)
#         regions = coords_df['region'].unique()
        
#         for region in regions:
#             print(f"Generating scatterplot for region {region}")
#             try:
#                 create_registration_scatterplots(
#                     timepoint_dir=tp_dir,
#                     original_coords_dir=original_coords_dir,
#                     output_dir=tp_scatterplots_dir,
#                     region=region
#                 )
#             except Exception as e:
#                 print(f"Error generating scatterplot for timepoint {timepoint}, region {region}: {e}")
#                 continue

# def visualize_registration(
#     grid: pd.DataFrame,
#     stats: Dict,
#     output_path: Union[str, Path] = None
# ) -> None:
#     """Visualize registration results.
    
#     Args:
#         grid: DataFrame containing registration grid information
#         stats: Dictionary containing registration statistics
#         output_path: Optional path to save the visualization
#     """
#     fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
#     # Plot translations
#     ax = axes[0]
#     for direction in ['left', 'top']:
#         if f'{direction}_x' in grid.columns and f'{direction}_y' in grid.columns:
#             valid_mask = grid[direction].notna()
#             if valid_mask.any():
#                 ax.quiver(
#                     grid.loc[valid_mask, f'{direction}_x'],
#                     grid.loc[valid_mask, f'{direction}_y'],
#                     grid.loc[valid_mask, f'{direction}_x_second'] - grid.loc[valid_mask, f'{direction}_x'],
#                     grid.loc[valid_mask, f'{direction}_y_second'] - grid.loc[valid_mask, f'{direction}_y'],
#                     scale=1.0,
#                     label=direction
#                 )
    
#     ax.set_title('Translation Vectors')
#     ax.legend()
#     ax.set_aspect('equal')
    
#     # Plot correlation scores
#     ax = axes[1]
#     for direction in ['left', 'top']:
#         if f'{direction}_corr' in grid.columns:
#             valid_mask = grid[direction].notna()
#             if valid_mask.any():
#                 ax.scatter(
#                     grid.loc[valid_mask, f'{direction}_x'],
#                     grid.loc[valid_mask, f'{direction}_y'],
#                     c=grid.loc[valid_mask, f'{direction}_corr'],
#                     cmap='viridis',
#                     label=direction
#                 )
    
#     ax.set_title('Correlation Scores')
#     ax.legend()
#     ax.set_aspect('equal')
    
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path)
#     else:
#         plt.show()
#     plt.close()

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Generate registration scatterplots for microscope image tiles.')
#     parser.add_argument('base_dir', type=str, help='Base directory containing timepoint subdirectories')
    
#     args = parser.parse_args()
    
#     try:
#         generate_all_registration_scatterplots(args.base_dir)
#         print("Successfully generated all scatterplots!")
#     except Exception as e:
#         print(f"Error: {e}")
#         exit(1)
