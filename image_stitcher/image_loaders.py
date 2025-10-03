"""Image loaders for different microscopy file formats.

This module provides a unified interface for loading images from various
file formats used in microscopy, including individual TIFF files, multi-page
TIFF files, and OME-TIFF files.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple
import re

import numpy as np


class ImageLoader(ABC):
    """Abstract base class for image loaders.
    
    Provides a unified interface for loading microscopy images from different
    file formats. Each loader handles format-specific details while presenting
    a consistent API.
    """
    
    def __init__(self, filepath: Path):
        """Initialize loader with file path.
        
        Args:
            filepath: Path to the image file
        """
        self.filepath = Path(filepath)
        self._metadata: Optional[Dict] = None
        
    @property
    @abstractmethod
    def metadata(self) -> Dict:
        """Get image metadata (lazy loaded).
        
        Returns:
            Dictionary containing:
                - num_channels: Number of channels
                - num_z: Number of Z-slices
                - shape: Image shape (height, width)
                - dtype: Data type
                - channel_names: Optional list of channel names
        """
        pass
    
    @abstractmethod
    def read_slice(self, channel: int = 0, z: int = 0) -> np.ndarray:
        """Read a single 2D image slice.
        
        Args:
            channel: Channel index (0-based)
            z: Z-slice index (0-based)
            
        Returns:
            2D numpy array of shape (height, width)
        """
        pass
    
    def read_channel_stack(self, channel: int = 0) -> np.ndarray:
        """Read all Z-slices for a channel.
        
        Args:
            channel: Channel index (0-based)
            
        Returns:
            3D numpy array of shape (num_z, height, width)
        """
        num_z = self.metadata['num_z']
        slices = [self.read_slice(channel, z) for z in range(num_z)]
        return np.stack(slices, axis=0)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.filepath.name}')"


class IndividualFileLoader(ImageLoader):
    """Loader for individual TIFF files (one file per channel/Z-slice)."""
    
    @property
    def metadata(self) -> Dict:
        """Get metadata from individual file."""
        if self._metadata is None:
            try:
                import tifffile
                with tifffile.TiffFile(self.filepath) as tif:
                    page = tif.pages[0]
                    self._metadata = {
                        'num_channels': 1,  # Individual files are single channel
                        'num_z': 1,         # Individual files are single Z-slice
                        'shape': page.shape,
                        'dtype': page.dtype,
                        'channel_names': None
                    }
            except ImportError:
                # Fallback to PIL if tifffile not available
                from PIL import Image
                with Image.open(self.filepath) as img:
                    arr = np.array(img)
                    self._metadata = {
                        'num_channels': 1,
                        'num_z': 1,
                        'shape': arr.shape,
                        'dtype': arr.dtype,
                        'channel_names': None
                    }
        return self._metadata
    
    def read_slice(self, channel: int = 0, z: int = 0) -> np.ndarray:
        """Read the single image from file."""
        if channel != 0 or z != 0:
            raise ValueError(
                f"Individual file loader only supports channel=0, z=0. "
                f"Got channel={channel}, z={z}"
            )
        
        try:
            import tifffile
            return tifffile.imread(self.filepath)
        except ImportError:
            from PIL import Image
            with Image.open(self.filepath) as img:
                return np.array(img)


class MultiPageTiffLoader(ImageLoader):
    """Loader for multi-page TIFF files.
    
    Multi-page TIFFs store multiple Z-slices as separate pages in one file.
    Each page represents a 2D image at a different Z position.
    """
    
    @property
    def metadata(self) -> Dict:
        """Get metadata from multi-page TIFF."""
        if self._metadata is None:
            try:
                import tifffile
                with tifffile.TiffFile(self.filepath) as tif:
                    self._metadata = {
                        'num_channels': 1,  # Multi-page TIFFs are single channel
                        'num_z': len(tif.pages),
                        'shape': tif.pages[0].shape,
                        'dtype': tif.pages[0].dtype,
                        'channel_names': None
                    }
            except ImportError:
                from PIL import Image
                with Image.open(self.filepath) as img:
                    arr = np.array(img)
                    self._metadata = {
                        'num_channels': 1,
                        'num_z': img.n_frames,
                        'shape': arr.shape,
                        'dtype': arr.dtype,
                        'channel_names': None
                    }
        return self._metadata
    
    def read_slice(self, channel: int = 0, z: int = 0) -> np.ndarray:
        """Read a specific Z-slice from multi-page TIFF."""
        if channel != 0:
            raise ValueError(
                f"Multi-page TIFF loader only supports channel=0. "
                f"Got channel={channel}"
            )
        
        num_z = self.metadata['num_z']
        if not (0 <= z < num_z):
            raise ValueError(f"Z-slice {z} out of range [0, {num_z})")
        
        try:
            import tifffile
            with tifffile.TiffFile(self.filepath) as tif:
                return tif.pages[z].asarray()
        except ImportError:
            from PIL import Image
            with Image.open(self.filepath) as img:
                img.seek(z)
                return np.array(img)


class OMETiffLoader(ImageLoader):
    """Loader for OME-TIFF files.
    
    OME-TIFF files contain multi-dimensional data with embedded OME-XML metadata.
    Structure: Multiple pages ordered as C (channels) × Z (slices).
    Page index = channel_index * num_z + z_index
    """
    
    @property
    def metadata(self) -> Dict:
        """Parse OME-XML metadata."""
        if self._metadata is None:
            import tifffile
            with tifffile.TiffFile(self.filepath) as tif:
                # Use tifffile's built-in OME metadata parsing
                if not tif.is_ome:
                    raise ValueError(f"File {self.filepath} is not a valid OME-TIFF")
                
                # Get first series (usually only one series per file)
                series = tif.series[0]
                
                # Parse shape based on axes
                # Common axes: 'TCZYX', 'CZYX', 'ZYX', 'YX'
                axes = series.axes
                shape_dict = dict(zip(axes, series.shape))
                
                num_channels = shape_dict.get('C', 1)
                num_z = shape_dict.get('Z', 1)
                num_t = shape_dict.get('T', 1)
                height = shape_dict.get('Y', series.shape[-2])
                width = shape_dict.get('X', series.shape[-1])
                
                # Try to get channel names from OME metadata
                channel_names = None
                if tif.ome_metadata:
                    channel_names = self._extract_channel_names(tif.ome_metadata)
                
                self._metadata = {
                    'num_channels': num_channels,
                    'num_z': num_z,
                    'num_t': num_t,
                    'shape': (height, width),
                    'dtype': series.dtype,
                    'channel_names': channel_names,
                    'axes': axes,
                    'full_shape': series.shape
                }
        
        return self._metadata
    
    def read_slice(self, channel: int = 0, z: int = 0) -> np.ndarray:
        """Read specific channel and Z-slice from OME-TIFF.
        
        Args:
            channel: Channel index (0-based)
            z: Z-slice index (0-based)
            
        Returns:
            2D numpy array of shape (height, width)
        """
        meta = self.metadata
        
        if not (0 <= channel < meta['num_channels']):
            raise ValueError(
                f"Channel {channel} out of range [0, {meta['num_channels']})"
            )
        if not (0 <= z < meta['num_z']):
            raise ValueError(
                f"Z-slice {z} out of range [0, {meta['num_z']})"
            )
        
        import tifffile
        with tifffile.TiffFile(self.filepath) as tif:
            series = tif.series[0]
            axes = meta['axes']
            
            # Build index based on axes order
            # Handle different axis orderings
            if 'T' in axes:
                # Multi-timepoint: we only read T=0 for now
                if axes == 'TCZYX':
                    img = series.asarray()[0, channel, z, :, :]
                elif axes == 'TZCYX':
                    # Time, Z, Channels, Y, X - swap C and Z order
                    img = series.asarray()[0, z, channel, :, :]
                elif axes == 'TZYX':
                    if channel != 0:
                        raise ValueError(f"No channel dimension, channel must be 0")
                    img = series.asarray()[0, z, :, :]
                elif axes == 'TCYX':
                    if z != 0:
                        raise ValueError(f"No Z dimension, z must be 0")
                    img = series.asarray()[0, channel, :, :]
                elif axes == 'TYX':
                    if channel != 0 or z != 0:
                        raise ValueError(f"2D+T image, channel and z must be 0")
                    img = series.asarray()[0, :, :]
                else:
                    raise NotImplementedError(f"Axis order {axes} not yet supported")
            else:
                # Single timepoint
                if axes == 'CZYX':
                    img = series.asarray()[channel, z, :, :]
                elif axes == 'ZYX':
                    if channel != 0:
                        raise ValueError(f"No channel dimension, channel must be 0")
                    img = series.asarray()[z, :, :]
                elif axes == 'CYX':
                    if z != 0:
                        raise ValueError(f"No Z dimension, z must be 0")
                    img = series.asarray()[channel, :, :]
                elif axes == 'YX':
                    if channel != 0 or z != 0:
                        raise ValueError(f"2D image, channel and z must be 0")
                    img = series.asarray()
                else:
                    raise NotImplementedError(f"Axis order {axes} not yet supported")
            
            return img
    
    def _extract_channel_names(self, ome_xml: str) -> Optional[list]:
        """Extract channel names from OME-XML metadata.
        
        Args:
            ome_xml: OME-XML metadata string
            
        Returns:
            List of channel names or None if not found
        """
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(ome_xml)
            
            # OME namespace
            ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
            
            # Find all channel elements
            channels = root.findall('.//ome:Channel', ns)
            if channels:
                names = []
                for ch in channels:
                    name = ch.get('Name')
                    if name:
                        names.append(name)
                    else:
                        # Fallback to ID
                        names.append(ch.get('ID', f'Channel_{len(names)}'))
                return names if names else None
        except Exception:
            # If parsing fails, return None
            return None


def parse_ome_tiff_filename(filename: str) -> Optional[Dict[str, any]]:
    """Parse OME-TIFF filename to extract region and FOV information.
    
    Expected patterns:
        - 'A1_0.ome.tif' → {'region': 'A1', 'fov': 0}
        - 'C3_15.ome.tif' → {'region': 'C3', 'fov': 15}
        - 'Well_A_Pos_0.ome.tif' → Various patterns
    
    Args:
        filename: Name of the OME-TIFF file
        
    Returns:
        Dictionary with 'region' and 'fov' keys, or None if pattern doesn't match
    """
    # Pattern 1: RegionName_FOV.ome.tif (e.g., A1_0.ome.tif)
    match = re.match(r'([A-Z]\d+)_(\d+)\.ome\.tiff?$', filename, re.IGNORECASE)
    if match:
        return {
            'region': match.group(1),
            'fov': int(match.group(2))
        }
    
    # Pattern 2: Region_FOV.ome.tif with longer names
    match = re.match(r'([A-Za-z0-9]+)_(\d+)\.ome\.tiff?$', filename, re.IGNORECASE)
    if match:
        return {
            'region': match.group(1),
            'fov': int(match.group(2))
        }
    
    return None


def detect_image_format(directory: Path) -> str:
    """Detect the image format used in a directory.
    
    Checks for:
    1. OME-TIFF files in directory or in parent's ome_tiff/ subdirectory
    2. Multi-page TIFF files
    3. Individual TIFF files
    
    Args:
        directory: Directory containing image files (usually a timepoint directory like '0/')
        
    Returns:
        One of: 'ome_tiff', 'multipage_tiff', 'individual_files'
    """
    directory = Path(directory)
    
    # Check for new OME-TIFF structure: parent/ome_tiff/ directory
    parent_ome_dir = directory.parent / "ome_tiff"
    if parent_ome_dir.exists() and parent_ome_dir.is_dir():
        ome_files = list(parent_ome_dir.glob("*.ome.tif")) + list(parent_ome_dir.glob("*.ome.tiff"))
        if ome_files:
            return 'ome_tiff'
    
    # Check for OME-TIFF files in current directory (legacy structure)
    ome_tiff_files = list(directory.glob("*.ome.tif")) + list(directory.glob("*.ome.tiff"))
    if ome_tiff_files:
        return 'ome_tiff'
    
    # Check for multi-page TIFF files
    tiff_files = list(directory.glob("*.tif")) + list(directory.glob("*.tiff"))
    if tiff_files:
        # Sample first file to check if multi-page
        try:
            import tifffile
            with tifffile.TiffFile(tiff_files[0]) as tif:
                if len(tif.pages) > 1:
                    return 'multipage_tiff'
        except ImportError:
            from PIL import Image
            with Image.open(tiff_files[0]) as img:
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    return 'multipage_tiff'
        
        return 'individual_files'
    
    # Default to individual files
    return 'individual_files'


def create_image_loader(filepath: Path, format_hint: Optional[str] = None) -> ImageLoader:
    """Factory function to create appropriate image loader.
    
    Args:
        filepath: Path to the image file
        format_hint: Optional format hint ('ome_tiff', 'multipage_tiff', 'individual_files')
        
    Returns:
        Appropriate ImageLoader instance
    """
    filepath = Path(filepath)
    
    # Auto-detect format if not provided
    if format_hint is None:
        if filepath.suffix.lower() in ['.ome.tif', '.ome.tiff']:
            format_hint = 'ome_tiff'
        elif filepath.name.lower().endswith('.ome.tif') or filepath.name.lower().endswith('.ome.tiff'):
            format_hint = 'ome_tiff'
        else:
            # Check if multi-page by loading
            try:
                import tifffile
                with tifffile.TiffFile(filepath) as tif:
                    format_hint = 'multipage_tiff' if len(tif.pages) > 1 else 'individual_files'
            except ImportError:
                from PIL import Image
                with Image.open(filepath) as img:
                    format_hint = 'multipage_tiff' if (hasattr(img, 'n_frames') and img.n_frames > 1) else 'individual_files'
    
    # Create loader based on format
    if format_hint == 'ome_tiff':
        return OMETiffLoader(filepath)
    elif format_hint == 'multipage_tiff':
        return MultiPageTiffLoader(filepath)
    else:
        return IndividualFileLoader(filepath)

