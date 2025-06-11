"""Tests for the registration module."""
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ..tile_registration import (
    register_tiles,
    extract_tile_indices,
    calculate_pixel_size_microns,
    update_stage_coordinates,
    read_coordinates_csv,
)

# Test data
@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    # Create 4 sample images in a 2x2 grid with significant overlap
    images = []
    for i in range(4):
        img = np.zeros((100, 100), dtype=np.uint16)
        # Add some unique pattern to each image with overlap
        img[30:70, 30:70] = 1000 + i * 100  # Larger central region for overlap
        images.append(img)
    return np.array(images)

@pytest.fixture
def sample_coordinates():
    """Create sample coordinates DataFrame."""
    data = {
        'fov': [0, 1, 2, 3],
        'x (mm)': [0.0, 1.0, 0.0, 1.0],
        'y (mm)': [0.0, 0.0, 1.0, 1.0],
    }
    return pd.DataFrame(data)

def test_extract_tile_indices(sample_coordinates):
    """Test tile index extraction."""
    filenames = [
        'test_0_0_image.tif',
        'test_1_0_image.tif',
        'test_2_0_image.tif',
        'test_3_0_image.tif'
    ]
    rows, cols, fname_map = extract_tile_indices(filenames, sample_coordinates)
    
    assert len(rows) == len(filenames)
    assert len(cols) == len(filenames)
    assert len(fname_map) == len(filenames)
    
    # Check row assignments
    assert rows[0] == rows[1]  # First row
    assert rows[2] == rows[3]  # Second row
    assert rows[0] != rows[2]  # Different rows
    
    # Check column assignments
    assert cols[0] == cols[2]  # First column
    assert cols[1] == cols[3]  # Second column
    assert cols[0] != cols[1]  # Different columns

def test_calculate_pixel_size_microns(sample_coordinates):
    """Test pixel size calculation."""
    grid = pd.DataFrame({
        'left': [None, 1, None, 2],
        'top': [None, None, 3, 4],
        'left_x': [0, 10, 0, 10],
        'left_y': [0, 0, 0, 0],
        'top_x': [0, 0, 0, 0],
        'top_y': [0, 0, 10, 10],
        'x_pos': [0, 100, 0, 100],
        'y_pos': [0, 0, 100, 100],
    })
    
    filenames = ['img_0_0.tif', 'img_1_0.tif', 'img_0_1.tif', 'img_1_1.tif']
    fname_to_idx = {fname: i for i, fname in enumerate(filenames)}
    
    pixel_size = calculate_pixel_size_microns(
        grid,
        sample_coordinates,
        fname_to_idx,
        filenames
    )
    
    assert isinstance(pixel_size, float)
    assert pixel_size > 0

def test_read_coordinates_csv():
    """Test reading coordinates from CSV."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        f.write(b'fov,x (mm),y (mm)\n0,0.0,0.0\n1,1.0,0.0\n')
        f.flush()
        
        df = read_coordinates_csv(f.name)
        
        assert isinstance(df, pd.DataFrame)
        assert 'fov' in df.columns
        assert 'x (mm)' in df.columns
        assert 'y (mm)' in df.columns
        assert len(df) == 2
        
    os.unlink(f.name)

def test_update_stage_coordinates(sample_coordinates):
    """Test stage coordinate updates."""
    grid = pd.DataFrame({
        'left': [None, 1, None, 2],
        'top': [None, None, 3, 4],
        'left_x': [0, 10, 0, 10],
        'left_y': [0, 0, 0, 0],
        'top_x': [0, 0, 0, 0],
        'top_y': [0, 0, 10, 10],
        'x_pos': [0, 100, 0, 100],
        'y_pos': [0, 0, 100, 100],
    })
    
    filenames = ['img_0_0.tif', 'img_1_0.tif', 'img_0_1.tif', 'img_1_1.tif']
    fname_to_idx = {fname: i for i, fname in enumerate(filenames)}
    
    updated_coords = update_stage_coordinates(
        grid,
        sample_coordinates,
        fname_to_idx,
        filenames,
        pixel_size_um=1.0
    )
    
    assert isinstance(updated_coords, pd.DataFrame)
    assert 'x (mm)' in updated_coords.columns
    assert 'y (mm)' in updated_coords.columns
    assert len(updated_coords) == len(sample_coordinates) 