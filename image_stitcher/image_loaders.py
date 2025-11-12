"""Image loaders for different microscopy file formats.

This module provides a unified interface for loading images from various
file formats used in microscopy, including individual TIFF files, multi-page
TIFF files, and OME-TIFF files.

DESIGN PRINCIPLE: Standardized Axes
------------------------------------
All loaders normalize their output to a consistent (Y, X) format for 2D images,
regardless of the underlying file format's axis ordering. This prevents bugs
caused by format-specific dimension ordering (e.g., OME-TIFF's (Z, C, Y, X) vs
single TIFF's (Y, X)). The complexity of different formats is handled once at
the loader boundary, not scattered throughout the codebase.

Key functions:
- read_slice(): Always returns (Y, X) array
- metadata['shape']: Always (height, width) tuple
- get_image_dimensions(): Convenience function for dimension queries
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import re
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _read_tiff_with_fallback(filepath: Path) -> np.ndarray:
    """Read TIFF file with tifffile, falling back to PIL if unavailable.
    
    Args:
        filepath: Path to TIFF file
        
    Returns:
        Numpy array of image data
    """
    try:
        import tifffile
        return tifffile.imread(filepath)
    except ImportError:
        from PIL import Image
        with Image.open(filepath) as img:
            return np.array(img)


def _get_tiff_metadata(filepath: Path) -> Dict:
    """Get basic TIFF metadata with tifffile, falling back to PIL if unavailable.
    
    Args:
        filepath: Path to TIFF file
        
    Returns:
        Dictionary with 'shape' and 'dtype' keys
    """
    try:
        import tifffile
        with tifffile.TiffFile(filepath) as tif:
            page = tif.pages[0]
            return {
                'shape': page.shape,
                'dtype': page.dtype,
                'num_pages': len(tif.pages)
            }
    except ImportError:
        from PIL import Image
        with Image.open(filepath) as img:
            arr = np.array(img)
            return {
                'shape': arr.shape,
                'dtype': arr.dtype,
                'num_pages': getattr(img, 'n_frames', 1)
            }


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
            meta = _get_tiff_metadata(self.filepath)
            self._metadata = {
                'num_channels': 1,  # Individual files are single channel
                'num_z': 1,         # Individual files are single Z-slice
                'shape': meta['shape'],
                'dtype': meta['dtype'],
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
        return _read_tiff_with_fallback(self.filepath)


class MultiPageTiffLoader(ImageLoader):
    """Loader for multi-page TIFF files.
    
    Multi-page TIFFs store multiple Z-slices as separate pages in one file.
    Each page represents a 2D image at a different Z position.
    """
    
    @property
    def metadata(self) -> Dict:
        """Get metadata from multi-page TIFF."""
        if self._metadata is None:
            meta = _get_tiff_metadata(self.filepath)
            self._metadata = {
                'num_channels': 1,  # Multi-page TIFFs are single channel
                'num_z': meta['num_pages'],
                'shape': meta['shape'],
                'dtype': meta['dtype'],
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
            import logging
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
                
                logging.info(f"OME-TIFF {self.filepath.name}: axes='{axes}', shape={series.shape}, shape_dict={shape_dict}")
                
                num_channels = shape_dict.get('C', 1)
                num_z = shape_dict.get('Z', 1)
                num_t = shape_dict.get('T', 1)
                height = shape_dict.get('Y', series.shape[-2])
                width = shape_dict.get('X', series.shape[-1])
                
                # Try to get channel names from OME metadata
                channel_names = None
                if tif.ome_metadata:
                    channel_names = self._extract_channel_names(tif.ome_metadata)
                
                logging.info(f"OME-TIFF {self.filepath.name}: detected {num_channels} channels, {num_z} z-slices. Channel names: {channel_names}")
                
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
    
    def read_slice(self, channel: int = 0, z: int = 0, t: int = 0) -> np.ndarray:
        """Read specific channel and Z-slice from OME-TIFF.
        
        Args:
            channel: Channel index (0-based)
            z: Z-slice index (0-based)
            t: Timepoint index (0-based, default=0)
            
        Returns:
            2D numpy array of shape (height, width)
        """
        meta = self.metadata
        
        # Validate indices
        if not (0 <= channel < meta['num_channels']):
            raise ValueError(
                f"Channel {channel} out of range [0, {meta['num_channels']})"
            )
        if not (0 <= z < meta['num_z']):
            raise ValueError(
                f"Z-slice {z} out of range [0, {meta['num_z']})"
            )
        if not (0 <= t < meta.get('num_t', 1)):
            raise ValueError(
                f"Timepoint {t} out of range [0, {meta.get('num_t', 1)})"
            )
        
        import tifffile
        with tifffile.TiffFile(self.filepath) as tif:
            series = tif.series[0]
            axes = meta['axes']
            
            # Build index tuple dynamically based on axes order
            index = []
            for axis in axes:
                if axis == 'T':
                    index.append(t)
                elif axis == 'C':
                    index.append(channel)
                elif axis == 'Z':
                    index.append(z)
                elif axis == 'Y':
                    index.append(slice(None))  # Full slice for Y
                elif axis == 'X':
                    index.append(slice(None))  # Full slice for X
                else:
                    raise ValueError(f"Unsupported axis '{axis}' in axes '{axes}'")
            
            return series.asarray()[tuple(index)]
    
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


# ============================================================================
# Channel Discovery Abstraction
# ============================================================================

from dataclasses import dataclass


@dataclass
class ChannelDescriptor:
    """Describes a loadable channel, format-agnostic.
    
    This abstraction allows uniform channel handling whether channels come from:
    - OME-TIFF (multiple channels in one file)
    - Individual files (one file per channel)
    - Multi-page TIFF (single channel, multiple Z)
    """
    index: int                      # Selection index for GUI/registration (0-based)
    name: str                       # Display name (e.g., "488nm", "DAPI", "Channel 0")
    wavelength_nm: Optional[int]    # Parsed wavelength if available
    loader: ImageLoader             # Loader instance for this channel
    channel_param: int              # Which channel to pass to loader.read_slice()
    z_param: int                    # Which z to pass to loader.read_slice()
    
    @property
    def is_fluorescence(self) -> bool:
        """Heuristic: is this likely a fluorescence channel?"""
        if self.wavelength_nm is not None:
            return True
        name_upper = self.name.upper()
        fluor_markers = ['GFP', 'RFP', 'DAPI', 'CY3', 'CY5', 'FITC', 'TRITC', 'HOECHST']
        return any(marker in name_upper for marker in fluor_markers)
    
    @property
    def is_brightfield(self) -> bool:
        """Heuristic: is this likely a brightfield channel?"""
        name_upper = self.name.upper()
        return 'BF' in name_upper or 'BRIGHTFIELD' in name_upper or 'BRIGHT_FIELD' in name_upper


class ChannelSource:
    """Discovers and provides access to channels for a FOV."""
    
    @staticmethod
    def _parse_wavelength_from_string(s: str) -> Optional[int]:
        """Extract wavelength in nm from string.
        
        Handles: '488nm', '561_nm', '405 nm', 'Channel_488nm'
        """
        match = re.search(r'(\d{3,4})\s*_?nm', s.lower())
        if match:
            return int(match.group(1))
        return None
    
    @staticmethod
    def _extract_channel_name(filename: str, fallback_index: int = 0) -> str:
        """Extract human-readable channel name from filename.
        
        Returns wavelength, dye name, or generic fallback.
        """
        filename_lower = filename.lower()
        
        # Wavelength patterns
        match = re.search(r'(\d{3,4})\s*_?nm', filename_lower)
        if match:
            return f"{match.group(1)}nm"
        
        # RGB channels
        match = re.search(r'_([RGB])\.tiff?$', filename, re.IGNORECASE)
        if match:
            return f"RGB_{match.group(1).upper()}"
        
        # Named fluorophores
        fluor_map = {
            'dapi': 'DAPI', 'hoechst': 'Hoechst',
            'gfp': 'GFP', 'fitc': 'FITC',
            'rfp': 'RFP', 'tritc': 'TRITC',
            'cy3': 'Cy3', 'cy5': 'Cy5',
        }
        for marker, name in fluor_map.items():
            if marker in filename_lower:
                return name
        
        # Brightfield
        if 'brightfield' in filename_lower or 'bright_field' in filename_lower:
            return 'BF'
        if re.search(r'\bbf\b', filename_lower):
            return 'BF'
        
        return f"Channel {fallback_index}"
    
    @staticmethod
    def discover(files: List[Path], region: str, fov: int, z_level: Optional[int] = None) -> List[ChannelDescriptor]:
        """Discover all channels for a specific FOV.
        
        Args:
            files: List of all image files
            region: Region identifier
            fov: FOV number
            z_level: Optional z-level (for OME-TIFF or individual files with z)
            
        Returns:
            List of ChannelDescriptor objects
        """
        from .registration.tile_registration import parse_filename_fov_info
        
        # Find files matching this FOV
        matching_files = []
        for path in files:
            fov_info = parse_filename_fov_info(path.name)
            if not fov_info:
                continue
            if fov_info.get('region') == region and fov_info.get('fov') == fov:
                matching_files.append((path, fov_info))
        
        if not matching_files:
            return []
        
        # Detect format
        first_path = matching_files[0][0]
        is_ome_tiff = first_path.name.lower().endswith(('.ome.tif', '.ome.tiff'))
        
        if is_ome_tiff:
            # OME-TIFF: One file, multiple channels inside
            loader = OMETiffLoader(first_path)
            meta = loader.metadata
            num_channels = meta['num_channels']
            channel_names = meta.get('channel_names') or []
            
            # Determine z index
            z_idx = z_level if z_level is not None else (meta['num_z'] // 2 if meta['num_z'] > 1 else 0)
            
            descriptors = []
            for ch_idx in range(num_channels):
                name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Channel {ch_idx}"
                wavelength = ChannelSource._parse_wavelength_from_string(name)
                
                descriptors.append(ChannelDescriptor(
                    index=ch_idx,
                    name=name,
                    wavelength_nm=wavelength,
                    loader=loader,
                    channel_param=ch_idx,
                    z_param=z_idx
                ))
            
            return descriptors
        
        else:
            # Individual files: Each file is a channel
            # Filter by z_level if specified
            channel_files = {}
            for path, fov_info in matching_files:
                file_z = fov_info.get('z_level')
                if z_level is not None and file_z != z_level:
                    continue
                
                # Extract channel name from filename
                ch_name = ChannelSource._extract_channel_name(path.name)
                channel_files[ch_name] = path
            
            if not channel_files:
                return []
            
            # Sort for consistent ordering
            sorted_channel_names = sorted(channel_files.keys())
            
            descriptors = []
            for idx, ch_name in enumerate(sorted_channel_names):
                filepath = channel_files[ch_name]
                loader = IndividualFileLoader(filepath)
                wavelength = ChannelSource._parse_wavelength_from_string(ch_name)
                
                descriptors.append(ChannelDescriptor(
                    index=idx,
                    name=ch_name,
                    wavelength_nm=wavelength,
                    loader=loader,
                    channel_param=0,  # Individual files are single-channel
                    z_param=0         # Individual files are single Z-slice
                ))
            
            return descriptors


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


def get_image_dimensions(filepath: Path) -> Tuple[int, int]:
    """Get standardized (height, width) dimensions from any image format.
    
    This function abstracts away the complexity of different file formats
    and dimension orderings, always returning (Y, X) dimensions.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Tuple of (height, width) in pixels
        
    Example:
        >>> height, width = get_image_dimensions(Path("data.ome.tiff"))
        >>> print(f"Image is {height}x{width} pixels")
    """
    loader = create_image_loader(filepath)
    metadata = loader.metadata
    height, width = metadata['shape']
    return height, width

