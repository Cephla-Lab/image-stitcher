"""Fixed registration module for manual tile alignment."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QMessageBox, QLabel
)
from PyQt5.QtGui import QPixmap, QPainter

from .tile_registration import (
    extract_tile_indices, read_coordinates_csv, load_single_image,
    parse_filename_fov_info, DEFAULT_FOV_RE, calculate_pixel_size_microns
)






@dataclass
class AlignmentTile:
    """A tile for manual alignment."""
    filename: str
    image: np.ndarray
    grid_row: int
    grid_col: int
    stage_x_mm: float
    stage_y_mm: float
    is_center: bool = False
    all_channels: Dict[str, np.ndarray] = None
    current_channel: str = "default"
    
    def __post_init__(self):
        if self.all_channels is None:
            self.all_channels = {"default": self.image}


class DraggableTileItem(QGraphicsPixmapItem):
    """Draggable tile item for alignment canvas."""
    
    def __init__(self, pixmap: QPixmap, tile: AlignmentTile):
        super().__init__(pixmap)
        self.tile = tile
        self.setFlag(QGraphicsPixmapItem.ItemIsMovable, not tile.is_center)
        self.setFlag(QGraphicsPixmapItem.ItemIsSelectable, True)
        self.initial_pos = QPointF(0, 0)
        
    def mousePressEvent(self, event):
        if not self.tile.is_center:
            self.initial_pos = self.pos()
        super().mousePressEvent(event)
    
    def cycle_channel(self):
        """Cycle to next available channel with contrast normalization."""
        if len(self.tile.all_channels) <= 1:
            return
            
        channels = list(self.tile.all_channels.keys())
        current_idx = channels.index(self.tile.current_channel)
        next_idx = (current_idx + 1) % len(channels)
        next_channel = channels[next_idx]
        
        # Switch to next channel
        self.tile.current_channel = next_channel
        self.tile.image = self.tile.all_channels[next_channel]
        
        # Update display with contrast normalization
        normalized_image = self.normalize_contrast(self.tile.image)
        self.setPixmap(self.array_to_pixmap(normalized_image))
        
        print(f"DEBUG: Switched to channel '{next_channel}' for {self.tile.filename}")
    
    def normalize_contrast(self, image: np.ndarray) -> np.ndarray:
        """Simple percentile-based contrast normalization (better than histogram equalization)."""
        if image.dtype != np.uint8:
            # Convert to 8-bit for display
            image_min, image_max = image.min(), image.max()
            if image_max > image_min:
                image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        
        # Simple percentile stretch - much better than histogram equalization for microscopy
        p2, p98 = np.percentile(image, [2, 98])
        if p98 > p2:
            image = np.clip((image - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        
        return image
    
    def array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap for display."""
        if array.ndim == 2:
            # Grayscale
            height, width = array.shape
            from PyQt5.QtGui import QImage
            qimg = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
        else:
            # RGB
            height, width = array.shape[:2]
            from PyQt5.QtGui import QImage
            qimg = QImage(array.data, width, height, width * 3, QImage.Format_RGB888)
            
        return QPixmap.fromImage(qimg)


class FixedRegistrationDialog(QDialog):
    """Minimal dialog for manual tile alignment."""
    
    # Class variable to persist positions across dialog instances
    _global_saved_positions = {}
    
    def __init__(self, alignment_tiles: List[AlignmentTile], coords_df: pd.DataFrame = None, image_directory: Path = None, parent=None):
        super().__init__(parent)
        self.alignment_tiles = alignment_tiles
        self.coords_df = coords_df
        self.image_directory = image_directory
        self.tile_items = {}
        
        # Refresh tile coordinates from current coordinates.csv
        self.refresh_tile_coordinates()
        
        self.setup_ui()
        self.load_tiles()
        
        # Enable keyboard shortcuts
        self.setFocusPolicy(Qt.StrongFocus)
    
    def refresh_tile_coordinates(self):
        """Update alignment tiles with current stage coordinates from coordinates.csv."""
        if self.coords_df is not None:
            for tile in self.alignment_tiles:
                # Find this tile by filename match (since coordinates.csv uses FOV, not row/col)
                from .tile_registration import parse_filename_fov_info
                fov_info = parse_filename_fov_info(tile.filename)
                
                if fov_info and 'fov' in fov_info:
                    fov_id = fov_info['fov']
                    # Look up by FOV in coordinates
                    tile_rows = self.coords_df[self.coords_df['fov'] == fov_id]
                    if not tile_rows.empty:
                        tile_row = tile_rows.iloc[0]
                        old_x, old_y = tile.stage_x_mm, tile.stage_y_mm
                        tile.stage_x_mm = tile_row['x (mm)']
                        tile.stage_y_mm = tile_row['y (mm)']
                        print(f"DEBUG: Refreshed tile {tile.filename} (grid {tile.grid_row},{tile.grid_col}) coordinates: "
                              f"({old_x:.3f},{old_y:.3f}) → ({tile.stage_x_mm:.3f},{tile.stage_y_mm:.3f}) mm")
        
    def setup_ui(self):
        self.setWindowTitle("Fixed Registration")
        self.setModal(True)
        self.resize(1000, 800)
        
        layout = QVBoxLayout(self)
        
        # Canvas
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(Qt.black)
        
        self.view = QGraphicsView(self.scene)
        # Start with RubberBandDrag for item selection/dragging
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        
        # Enable smooth zooming
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # Enable keyboard focus for the view
        self.view.setFocusPolicy(Qt.StrongFocus)
        
        # Enable mouse handling, wheel zoom, and keyboard events
        self.view.mousePressEvent = self.view_mouse_press_event
        self.view.mouseReleaseEvent = self.view_mouse_release_event
        self.view.wheelEvent = self.view_wheel_event
        self.view.keyPressEvent = self.view_key_press_event
        self.view.enterEvent = self.view_enter_event
        layout.addWidget(self.view)
        
        # Channel info and help
        info_layout = QHBoxLayout()
        
        self.channel_info_label = QLabel("Drag tiles to align")
        self.channel_info_label.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(self.channel_info_label)
        
        info_layout.addStretch()
        
        # Help tooltip button
        help_btn = QPushButton("?")
        help_btn.setFixedSize(25, 25)
        help_btn.setStyleSheet("QPushButton { border-radius: 12px; font-weight: bold; }")
        help_btn.clicked.connect(self.show_help)
        info_layout.addWidget(help_btn)
        
        layout.addLayout(info_layout)
        
        # Controls
        controls = QHBoxLayout()
        
        select_other_btn = QPushButton("Select Other Three")
        select_other_btn.clicked.connect(self.select_other_tiles)
        controls.addWidget(select_other_btn)
        
        fit_view_btn = QPushButton("Fit to View")
        fit_view_btn.clicked.connect(self.fit_to_view)
        controls.addWidget(fit_view_btn)
        
        controls.addStretch()
        
        set_shifts_btn = QPushButton("Set Fixed Shifts")
        set_shifts_btn.clicked.connect(self.set_fixed_shifts)
        controls.addWidget(set_shifts_btn)
        
        layout.addLayout(controls)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        from PyQt5.QtCore import Qt
        
        if event.key() == Qt.Key_Up or event.key() == Qt.Key_Down:
            # Arrow keys cycle channels
            self.cycle_all_channels()
            event.accept()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Enter key sets fixed shifts
            self.set_fixed_shifts()
            event.accept()
        elif event.key() == Qt.Key_Plus or (event.key() == Qt.Key_Equal and event.modifiers() & Qt.ControlModifier):
            # Cmd/Ctrl + Plus: Zoom in
            self.view.scale(1.25, 1.25)
            event.accept()
        elif event.key() == Qt.Key_Minus and event.modifiers() & Qt.ControlModifier:
            # Cmd/Ctrl + Minus: Zoom out
            self.view.scale(0.8, 0.8)
            event.accept()
        elif event.key() == Qt.Key_0 and event.modifiers() & Qt.ControlModifier:
            # Cmd/Ctrl + 0: Fit to view
            self.fit_to_view()
            event.accept()
        elif event.key() == Qt.Key_Space:
            # Spacebar: Toggle between pan and drag modes
            if self.view.dragMode() == QGraphicsView.RubberBandDrag:
                self.view.setDragMode(QGraphicsView.ScrollHandDrag)
                self.channel_info_label.setText("Pan mode - drag to pan canvas")
            else:
                self.view.setDragMode(QGraphicsView.RubberBandDrag)
                self.channel_info_label.setText("Drag tiles to align")
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def show_help(self):
        """Show help dialog with keyboard shortcuts."""
        help_text = """
<b>↑ ↓</b> &nbsp;&nbsp;&nbsp; Cycle wavelengths/channels<br>
<b>Enter</b> &nbsp;&nbsp; Set fixed shifts<br>
<b>Space</b> &nbsp;&nbsp; Toggle pan mode<br><br>
Drag tiles to align them visually
        """
        
        QMessageBox.information(self, "Controls Help", help_text.strip())
    
    def view_mouse_press_event(self, event):
        """Handle mouse events on canvas with proper panning."""
        from PyQt5.QtCore import Qt
        
        if event.button() == Qt.MiddleButton:
            # Middle-click: Enable panning
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            # Simulate left-click for panning to start
            fake_event = event
            fake_event.button = lambda: Qt.LeftButton
            QGraphicsView.mousePressEvent(self.view, fake_event)
        else:
            # Normal behavior for left/right clicks
            QGraphicsView.mousePressEvent(self.view, event)
    
    def view_mouse_release_event(self, event):
        """Handle mouse release events."""
        from PyQt5.QtCore import Qt
        
        if event.button() == Qt.MiddleButton:
            # Reset drag mode after middle-click panning
            self.view.setDragMode(QGraphicsView.RubberBandDrag)
        
        # Call original mouseReleaseEvent
        QGraphicsView.mouseReleaseEvent(self.view, event)
    
    def view_wheel_event(self, event):
        """Handle mouse wheel for smooth zooming."""
        # Zoom factor
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        # Set zoom limits
        current_scale = self.view.transform().m11()  # Get current scale
        max_scale = 10.0  # 10x zoom
        min_scale = 0.1   # 0.1x zoom (zoom out)
        
        # Determine zoom direction
        if event.angleDelta().y() > 0:
            # Zoom in
            if current_scale < max_scale:
                self.view.scale(zoom_in_factor, zoom_in_factor)
        else:
            # Zoom out
            if current_scale > min_scale:
                self.view.scale(zoom_out_factor, zoom_out_factor)
        
        event.accept()
    
    def view_key_press_event(self, event):
        """Forward keyboard events from view to dialog."""
        # Forward to the dialog's keyPressEvent
        self.keyPressEvent(event)
    
    def view_enter_event(self, event):
        """Give focus to view when mouse enters."""
        self.view.setFocus()
        QGraphicsView.enterEvent(self.view, event)
    
    def fit_to_view(self):
        """Fit all tiles to view with some padding."""
        if self.tile_items:
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            # Add some padding by zooming out slightly
            self.view.scale(0.9, 0.9)
    
    def cycle_all_channels(self):
        """Cycle all tiles to the next available channel simultaneously."""
        if not self.tile_items:
            return
        
        # Get all available channels from center tile (they should all have same channels)
        center_tile = next(t for t in self.alignment_tiles if t.is_center)
        if len(center_tile.all_channels) <= 1:
            return
        
        channels = list(center_tile.all_channels.keys())
        current_idx = channels.index(center_tile.current_channel)
        next_idx = (current_idx + 1) % len(channels)
        next_channel = channels[next_idx]
        
        # Update all tiles to the same channel
        for tile_item in self.tile_items.values():
            if next_channel in tile_item.tile.all_channels:
                tile_item.tile.current_channel = next_channel
                tile_item.tile.image = tile_item.tile.all_channels[next_channel]
                
                # Update display with contrast normalization
                normalized_image = tile_item.normalize_contrast(tile_item.tile.image)
                tile_item.setPixmap(tile_item.array_to_pixmap(normalized_image))
        
        # Update channel info label
        self.channel_info_label.setText(f"Current channel: {next_channel}")
        print(f"DEBUG: Switched all tiles to channel '{next_channel}'")
        
    def load_tiles(self):
        """Load tiles onto canvas with multi-channel support."""
        # Clear existing tiles from scene
        self.scene.clear()
        self.tile_items = {}
        
        center_tile = next(t for t in self.alignment_tiles if t.is_center)
        other_tiles = [t for t in self.alignment_tiles if not t.is_center]
        
        # Update channel info
        if center_tile.all_channels and len(center_tile.all_channels) > 1:
            channels = list(center_tile.all_channels.keys())
            self.channel_info_label.setText(f"Current channel: {center_tile.current_channel} | Available: {', '.join(channels)}")
        else:
            self.channel_info_label.setText("Single channel detected")
        
        # Place center tile at origin with contrast normalization
        center_item = DraggableTileItem(QPixmap(), center_tile)  # Temp pixmap
        normalized_image = center_item.normalize_contrast(center_tile.image)
        center_pixmap = center_item.array_to_pixmap(normalized_image)
        center_item = DraggableTileItem(center_pixmap, center_tile)
        center_item.setPos(0, 0)
        self.scene.addItem(center_item)
        self.tile_items['center'] = center_item
        
        # Place other tiles at approximate positions based on grid
        for i, tile in enumerate(other_tiles):
            # Apply contrast normalization for initial display
            temp_item = DraggableTileItem(QPixmap(), tile)  # Temp for normalization
            normalized_image = temp_item.normalize_contrast(tile.image)
            pixmap = temp_item.array_to_pixmap(normalized_image)
            item = DraggableTileItem(pixmap, tile)
            
            # Initial positioning based on grid relationship to center
            if tile.grid_row < center_tile.grid_row:  # Top neighbor
                item.setPos(0, -center_pixmap.height() * 0.8)
                self.tile_items['top'] = item
            elif tile.grid_col < center_tile.grid_col:  # Left neighbor  
                item.setPos(-center_pixmap.width() * 0.8, 0)
                self.tile_items['left'] = item
                
            self.scene.addItem(item)
        
        # Restore previously saved positions if they exist
        if not self.restore_saved_positions():
            # If no saved positions, fit view to scene with padding
            self.fit_to_view()
        
    def array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap."""
        # Normalize to 0-255
        if array.dtype != np.uint8:
            array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
        
        # Create QPixmap
        height, width = array.shape[:2]
        if len(array.shape) == 2:  # Grayscale
            from PyQt5.QtGui import QImage
            qimg = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
        else:  # RGB
            from PyQt5.QtGui import QImage
            qimg = QImage(array.data, width, height, width * 3, QImage.Format_RGB888)
            
        return QPixmap.fromImage(qimg)
    
    def select_other_tiles(self):
        """Select a random different triplet for alignment."""
        if self.coords_df is None or self.image_directory is None:
            QMessageBox.information(self, "Info", "Cannot select other triplets - missing data")
            return
            
        try:
            # Find a random valid triplet (different from current)
            new_triplet = find_random_triplet(self.coords_df, self.image_directory, exclude_center=self.get_current_center_position())
            if new_triplet:
                self.alignment_tiles = new_triplet
                
                # Clear current scene and reload with new tiles
                self.scene.clear()
                self.tile_items.clear()
                self.load_tiles()
                
                print(f"DEBUG: Switched to new triplet with center at grid({new_triplet[0].grid_row}, {new_triplet[0].grid_col})")
            else:
                QMessageBox.information(self, "Info", "No other valid triplets found")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to find other triplets: {e}")
    
    def get_current_center_position(self) -> Tuple[int, int]:
        """Get current center tile's grid position."""
        center_tile = next(t for t in self.alignment_tiles if t.is_center)
        return (center_tile.grid_row, center_tile.grid_col)
        
    def calculate_shifts(self) -> Tuple[float, float]:
        """Calculate shifts using both displacement vectors from user's visual alignment."""
        center_tile = next(t for t in self.alignment_tiles if t.is_center)
        left_tile = next((t for t in self.alignment_tiles if t.grid_row == center_tile.grid_row and t.grid_col == center_tile.grid_col - 1), None)
        top_tile = next((t for t in self.alignment_tiles if t.grid_row == center_tile.grid_row - 1 and t.grid_col == center_tile.grid_col), None)
        
        # Collect displacement vectors from user's visual alignment
        displacement_vectors = []
        pixel_size_um = 0.325  # Will be refined with actual calculation
        
        # Get center position
        center_item = self.tile_items['center']
        center_pos = center_item.pos()
        
        # Calculate displacement vector from left neighbor
        if left_tile and 'left' in self.tile_items:
            left_item = self.tile_items['left']
            left_pos = left_item.pos()
            
            # Expected stage coordinate differences (mm)
            expected_stage_diff_x = center_tile.stage_x_mm - left_tile.stage_x_mm
            expected_stage_diff_y = center_tile.stage_y_mm - left_tile.stage_y_mm
            
            # Expected pixel differences based on stage coordinates
            expected_pixel_diff_x = (expected_stage_diff_x * 1000.0) / pixel_size_um
            expected_pixel_diff_y = (expected_stage_diff_y * 1000.0) / pixel_size_um
            
            # Current pixel differences (user's visual alignment)
            current_pixel_diff_x = center_pos.x() - left_pos.x()
            current_pixel_diff_y = center_pos.y() - left_pos.y()
            
            # Displacement vector = current - expected
            dx_left = current_pixel_diff_x - expected_pixel_diff_x
            dy_left = current_pixel_diff_y - expected_pixel_diff_y
            
            displacement_vectors.append((dx_left, dy_left))
            
            print(f"DEBUG Left neighbor displacement: ({dx_left:.1f}, {dy_left:.1f})px")
            
        # Calculate displacement vector from top neighbor
        if top_tile and 'top' in self.tile_items:
            top_item = self.tile_items['top']
            top_pos = top_item.pos()
            
            # Expected stage coordinate differences (mm)
            expected_stage_diff_x = center_tile.stage_x_mm - top_tile.stage_x_mm
            expected_stage_diff_y = center_tile.stage_y_mm - top_tile.stage_y_mm
            
            # Expected pixel differences based on stage coordinates
            expected_pixel_diff_x = (expected_stage_diff_x * 1000.0) / pixel_size_um
            expected_pixel_diff_y = (expected_stage_diff_y * 1000.0) / pixel_size_um
            
            # Current pixel differences (user's visual alignment)
            current_pixel_diff_x = center_pos.x() - top_pos.x()
            current_pixel_diff_y = center_pos.y() - top_pos.y()
            
            # Displacement vector = current - expected
            dx_top = current_pixel_diff_x - expected_pixel_diff_x
            dy_top = current_pixel_diff_y - expected_pixel_diff_y
            
            displacement_vectors.append((dx_top, dy_top))
            
            print(f"DEBUG Top neighbor displacement: ({dx_top:.1f}, {dy_top:.1f})px")
        
        # Average the displacement vectors (user's visual alignment is ground truth)
        if displacement_vectors:
            h_shift_px = sum(dx for dx, dy in displacement_vectors) / len(displacement_vectors)
            v_shift_px = sum(dy for dx, dy in displacement_vectors) / len(displacement_vectors)
            
            print(f"DEBUG Final averaged shifts: h_shift={h_shift_px:.1f}px, v_shift={v_shift_px:.1f}px")
            
            # Calculate consistency metric for user feedback
            if len(displacement_vectors) > 1:
                dx_diff = displacement_vectors[0][0] - displacement_vectors[1][0]
                dy_diff = displacement_vectors[0][1] - displacement_vectors[1][1]
                consistency_error = np.sqrt(dx_diff**2 + dy_diff**2)
                print(f"DEBUG Alignment consistency: {consistency_error:.1f}px difference between vectors")
        else:
            h_shift_px = 0.0
            v_shift_px = 0.0
            print("DEBUG: No neighbors available for shift calculation")
            
        return h_shift_px, v_shift_px
        
    def set_fixed_shifts(self):
        """Calculate and apply fixed shifts."""
        h_shift_px, v_shift_px = self.calculate_shifts()
        
        # Store shifts for later use
        self.h_shift_px = h_shift_px
        self.v_shift_px = v_shift_px
        
        # Show confirmation
        result = QMessageBox.question(
            self,
            "Apply Fixed Shifts",
            f"Apply these shifts to coordinates?\n\n"
            f"Horizontal: {h_shift_px:.1f} px\n"
            f"Vertical: {v_shift_px:.1f} px",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Store current positions for future restoration
            self.save_current_positions()
            self.accept()
    
    def save_current_positions(self):
        """Save current tile positions for future restoration (persists across dialog instances)."""
        center_tile = next(t for t in self.alignment_tiles if t.is_center)
        triplet_key = f"{center_tile.grid_row}_{center_tile.grid_col}"
        
        FixedRegistrationDialog._global_saved_positions[triplet_key] = {}
        for role, item in self.tile_items.items():
            FixedRegistrationDialog._global_saved_positions[triplet_key][role] = (item.pos().x(), item.pos().y())
        print(f"DEBUG: Saved tile positions for triplet {triplet_key}: {FixedRegistrationDialog._global_saved_positions[triplet_key]}")
    
    def restore_saved_positions(self):
        """Restore previously saved tile positions for this triplet."""
        center_tile = next(t for t in self.alignment_tiles if t.is_center)
        triplet_key = f"{center_tile.grid_row}_{center_tile.grid_col}"
        
        if triplet_key in FixedRegistrationDialog._global_saved_positions:
            saved_positions = FixedRegistrationDialog._global_saved_positions[triplet_key]
            for role, item in self.tile_items.items():
                if role in saved_positions:
                    x, y = saved_positions[role]
                    item.setPos(x, y)
            print(f"DEBUG: Restored tile positions for triplet {triplet_key}: {saved_positions}")
            return True
        return False
    


def get_channel_color(channel_name: str) -> int:
    """Compute the color for display of a given channel name (from grid_viewer_gui.py)."""
    color_map = {
        "405": 0x0000FF,  # Blue
        "488": 0x00FF00,  # Green
        "561": 0xFFCF00,  # Yellow
        "638": 0xFF0000,  # Red
        "730": 0x770000,  # Dark Red
        "_B": 0x0000FF,  # Blue
        "_G": 0x00FF00,  # Green
        "_R": 0xFF0000,  # Red
    }
    
    # Check for direct matches
    for pattern, color in color_map.items():
        if pattern in channel_name:
            return color
    
    # Default color for unknown channels
    return 0xFFFFFF  # White


def detect_channel_from_filename(filename: str) -> str:
    """Detect channel using official colormap patterns."""
    # First check for official colormap patterns (exact matches)
    official_patterns = ["405", "488", "561", "638", "730", "_B", "_G", "_R"]
    for pattern in official_patterns:
        if pattern in filename:
            return pattern
    
    filename_lower = filename.lower()
    
    # Extended patterns for common microscopy channels
    if 'dapi' in filename_lower or 'hoechst' in filename_lower:
        return '405'
    elif 'gfp' in filename_lower or 'fitc' in filename_lower:
        return '488'
    elif 'rfp' in filename_lower or 'texas' in filename_lower:
        return '561'
    elif 'cy5' in filename_lower or '647' in filename:
        return '638'
    elif 'bf' in filename_lower or 'bright' in filename_lower:
        return 'BF'
    else:
        return 'Unknown'


def find_random_triplet(coords_df: pd.DataFrame, image_directory: Path, exclude_center: Tuple[int, int] = None, max_attempts: int = 20) -> Optional[List[AlignmentTile]]:
    """Find a random VALID triplet efficiently without pre-computing all triplets.
    
    A VALID triplet means:
    1. Center tile has both left AND top neighbors in the grid
    2. All 3 images load successfully  
    3. All 3 tiles have valid stage coordinates
    4. Center is different from excluded position (if specified)
    """
    import random
    
    # Get all TIFF files
    tiff_files = list(image_directory.glob("*.tiff"))
    if not tiff_files:
        subdir = image_directory / "0"
        if subdir.exists():
            tiff_files = list(subdir.glob("*.tiff"))
    
    if not tiff_files:
        return None
    
    # Use first region found
    region = None
    for f in tiff_files:
        fov_info = parse_filename_fov_info(f.name)
        if fov_info and 'region' in fov_info:
            region = fov_info['region']
            break
    
    if not region:
        return None
    
    # Filter files for this region
    region_files = []
    for f in tiff_files:
        fov_info = parse_filename_fov_info(f.name)
        if fov_info and fov_info.get('region') == region:
            region_files.append(f)
    
    # Extract grid positions
    filenames = [f.name for f in region_files]
    region_coords = coords_df[coords_df['region'] == region].copy()
    rows, cols, filename_to_index = extract_tile_indices(filenames, region_coords)
    
    # Create master grid
    master_grid = pd.DataFrame({
        "col": cols,
        "row": rows,
    }, index=np.arange(len(cols)))
    
    # Handle duplicate positions efficiently
    unique_positions = {}
    for idx, (col, row) in enumerate(zip(master_grid['col'], master_grid['row'])):
        pos = (col, row)
        if pos not in unique_positions:
            unique_positions[pos] = idx
    
    # Create neighbor relationships
    top_neighbors = []
    left_neighbors = []
    
    for idx, (col, row) in enumerate(zip(master_grid['col'], master_grid['row'])):
        top_pos = (col, row - 1)
        left_pos = (col - 1, row)
        
        top_neighbors.append(unique_positions.get(top_pos, pd.NA))
        left_neighbors.append(unique_positions.get(left_pos, pd.NA))
    
    master_grid['top'] = pd.array(top_neighbors, dtype='Int64')
    master_grid['left'] = pd.array(left_neighbors, dtype='Int64')
    
    # Find valid center candidates (have both neighbors)
    valid_centers = []
    for idx, row_data in master_grid.iterrows():
        left_neighbor = None if pd.isna(row_data['left']) else int(row_data['left'])
        top_neighbor = None if pd.isna(row_data['top']) else int(row_data['top'])
        
        if left_neighbor is not None and top_neighbor is not None:
            center_pos = (rows[idx], cols[idx])
            # Skip if this is the excluded center
            if exclude_center is None or center_pos != exclude_center:
                valid_centers.append((idx, left_neighbor, top_neighbor))
    
    if not valid_centers:
        print("DEBUG: No valid triplet candidates found (no tiles with both neighbors)")
        return None
    
    print(f"DEBUG: Found {len(valid_centers)} VALID triplet candidates, selecting random one...")
    
    # Try random candidates until we find one that loads successfully
    random.shuffle(valid_centers)
    
    for attempt in range(min(max_attempts, len(valid_centers))):
        idx, left_neighbor, top_neighbor = valid_centers[attempt]
        
        try:
            # First, group all files by grid position to find all channels per position
            position_files = {}
            for i, (file_path, row, col) in enumerate(zip(region_files, rows, cols)):
                position = (row, col)
                if position not in position_files:
                    position_files[position] = []
                position_files[position].append((file_path, i))
            
            # Load images for this triplet with all channels
            tiles = []
            for tile_idx, tile_type in [(idx, 'center'), (top_neighbor, 'top'), (left_neighbor, 'left')]:
                position = (rows[tile_idx], cols[tile_idx])
                files_at_position = position_files[position]
                
                # Load all channels for this position
                all_channels = {}
                primary_image = None
                primary_filename = None
                
                for file_path, file_idx in files_at_position:
                    channel = detect_channel_from_filename(file_path.name)
                    _, image = load_single_image(file_path)
                    if image is not None:
                        all_channels[channel] = image
                        if primary_image is None:  # Use first successful load as primary
                            primary_image = image
                            primary_filename = file_path.name
                
                if primary_image is None:
                    break
                
                # Get coordinates using primary filename
                coord_idx = filename_to_index[primary_filename]
                stage_x = region_coords.loc[coord_idx, 'x (mm)']
                stage_y = region_coords.loc[coord_idx, 'y (mm)']
                
                # Determine primary channel using official colormap priority
                channel_priority = ['BF', '405', '488', '561', '638', '730', '_B', '_G', '_R']
                primary_channel = next((ch for ch in channel_priority if ch in all_channels), 
                                     list(all_channels.keys())[0])
                
                tile = AlignmentTile(
                    filename=primary_filename,
                    image=all_channels[primary_channel],
                    grid_row=rows[tile_idx],
                    grid_col=cols[tile_idx],
                    stage_x_mm=stage_x,
                    stage_y_mm=stage_y,
                    is_center=(tile_type == 'center'),
                    all_channels=all_channels,
                    current_channel=primary_channel
                )
                tiles.append(tile)
            
            if len(tiles) == 3:
                center_channels = list(tiles[0].all_channels.keys())
                print(f"DEBUG: Found random VALID triplet (attempt {attempt+1}): center at grid({rows[idx]}, {cols[idx]}) with channels {center_channels}")
                return tiles
                
        except Exception as e:
            print(f"DEBUG: Failed to load triplet {attempt+1}: {e}")
            continue
    
    return None


def get_all_alignment_triplets(coords_df: pd.DataFrame, image_directory: Path) -> List[List[AlignmentTile]]:
    """Get all possible tile triplets for alignment."""
    
    # Get all TIFF files
    tiff_files = list(image_directory.glob("*.tiff"))
    if not tiff_files:
        # Try in subdirectory
        subdir = image_directory / "0"
        if subdir.exists():
            tiff_files = list(subdir.glob("*.tiff"))
    
    if not tiff_files:
        raise ValueError("No TIFF files found")
    
    # Use first region found
    region = None
    for f in tiff_files:
        fov_info = parse_filename_fov_info(f.name)
        if fov_info and 'region' in fov_info:
            region = fov_info['region']
            break
    
    if not region:
        raise ValueError("No region found in filenames")
    
    # Filter files for this region
    region_files = []
    for f in tiff_files:
        fov_info = parse_filename_fov_info(f.name)
        if fov_info and fov_info.get('region') == region:
            region_files.append(f)
    
    # Extract grid positions using existing logic
    filenames = [f.name for f in region_files]
    region_coords = coords_df[coords_df['region'] == region].copy()
    rows, cols, filename_to_index = extract_tile_indices(filenames, region_coords)
    
    # Use EXACT same grid logic as tile_registration.py
    print(f"DEBUG: Found {len(rows)} tiles in region {region}")
    
    # Create master grid exactly like tile_registration.py
    master_grid = pd.DataFrame({
        "col": cols,
        "row": rows,
    }, index=np.arange(len(cols)))
    
    # Show grid occupancy matrix EXACTLY like tile_registration.py
    print("\nDEBUG: Grid Occupancy Matrix:")
    print("=" * 50)
    max_row = max(rows)
    max_col = max(cols)
    print(f"Grid size: {max_row + 1} rows × {max_col + 1} columns")
    
    # Create occupancy matrix
    occupancy = {}
    for i, (row, col) in enumerate(zip(rows, cols)):
        if (row, col) not in occupancy:
            occupancy[(row, col)] = []
        occupancy[(row, col)].append(i)
    
    # Show occupancy conflicts
    conflicts = [(pos, indices) for pos, indices in occupancy.items() if len(indices) > 1]
    if conflicts:
        print(f"WARNING: Found {len(conflicts)} grid position conflicts:")
        for (row, col), indices in conflicts:
            print(f"  Position (row={row}, col={col}) has tiles: {indices}")
    
    # Show grid pattern (first 10x10) EXACTLY like tile_registration.py
    print("\nDEBUG: Grid Pattern (showing up to 10x10, 'X' = occupied, '.' = empty):")
    display_rows = min(10, max_row + 1)
    display_cols = min(10, max_col + 1)
    for r in range(display_rows):
        row_str = ""
        for c in range(display_cols):
            if (r, c) in occupancy:
                row_str += "X "
            else:
                row_str += ". "
        print(f"Row {r:2d}: {row_str}")
    
    # Handle duplicate grid positions by using only the first occurrence
    print("\nDEBUG: Handling duplicate grid positions for neighbor detection...")
    
    # Create a mapping that handles duplicates by taking the first occurrence
    unique_positions = {}
    for idx, (col, row) in enumerate(zip(master_grid['col'], master_grid['row'])):
        pos = (col, row)
        if pos not in unique_positions:
            unique_positions[pos] = idx
    
    print(f"DEBUG: Reduced {len(master_grid)} tiles to {len(unique_positions)} unique positions")
    
    # Create neighbor relationships using manual lookup (avoiding MultiIndex issues)
    top_neighbors = []
    left_neighbors = []
    
    for idx, (col, row) in enumerate(zip(master_grid['col'], master_grid['row'])):
        # Find top neighbor (row-1, same col)
        top_pos = (col, row - 1)
        if top_pos in unique_positions:
            top_neighbors.append(unique_positions[top_pos])
        else:
            top_neighbors.append(pd.NA)
            
        # Find left neighbor (same row, col-1)  
        left_pos = (col - 1, row)
        if left_pos in unique_positions:
            left_neighbors.append(unique_positions[left_pos])
        else:
            left_neighbors.append(pd.NA)
    
    master_grid['top'] = pd.array(top_neighbors, dtype='Int64')
    master_grid['left'] = pd.array(left_neighbors, dtype='Int64')
    
    print("DEBUG: Neighbor assignments (first 10 rows):")
    neighbor_debug = master_grid[['col', 'row', 'left', 'top']].head(10)
    print(neighbor_debug)
    
    # Find ALL tiles with both neighbors using the SAME logic as tile_registration.py
    all_triplets = []
    for idx, row_data in master_grid.iterrows():
        left_neighbor = None if pd.isna(row_data['left']) else int(row_data['left'])
        top_neighbor = None if pd.isna(row_data['top']) else int(row_data['top'])
        
        if left_neighbor is not None and top_neighbor is not None:
            print(f"DEBUG: Found triplet {len(all_triplets)+1}: center={idx}, left={left_neighbor}, top={top_neighbor}")
            
            # Load images and create tiles
            tiles = []
            for tile_idx, tile_type in [(idx, 'center'), (top_neighbor, 'top'), (left_neighbor, 'left')]:
                file_path = region_files[tile_idx]
                filename = file_path.name
                
                # Load image
                _, image = load_single_image(file_path)
                if image is None:
                    print(f"ERROR: Failed to load image {filename}")
                    break
                
                # Get coordinates
                coord_idx = filename_to_index[filename]
                stage_x = region_coords.loc[coord_idx, 'x (mm)']
                stage_y = region_coords.loc[coord_idx, 'y (mm)']
                
                tile = AlignmentTile(
                    filename=filename,
                    image=image,
                    grid_row=rows[tile_idx],
                    grid_col=cols[tile_idx],
                    stage_x_mm=stage_x,
                    stage_y_mm=stage_y,
                    is_center=(tile_type == 'center')
                )
                tiles.append(tile)
            
            if len(tiles) == 3:
                all_triplets.append(tiles)
    
    if not all_triplets:
        raise ValueError("No suitable tile triplets found")
    
    print(f"DEBUG: Found {len(all_triplets)} total triplets")
    return all_triplets


def select_alignment_tiles(coords_df: pd.DataFrame, image_directory: Path) -> List[AlignmentTile]:
    """Select first available tile triplet for alignment."""
    triplet = find_random_triplet(coords_df, image_directory, exclude_center=None, max_attempts=1)
    if triplet is None:
        raise ValueError("No suitable tile triplet found")
    return triplet


def calculate_actual_pixel_size(alignment_tiles: List[AlignmentTile], coords_df: pd.DataFrame) -> float:
    """Calculate actual pixel size using existing calculate_pixel_size_microns function."""
    
    # Create a minimal grid DataFrame for the existing function
    grid_data = []
    filename_to_index = {}
    filenames = []
    
    for i, tile in enumerate(alignment_tiles):
        # Add to grid (using dummy positions - the function calculates from stage coords)
        grid_data.append({
            'x_pos': i * 1000,  # Dummy pixel positions
            'y_pos': 0,
            'left': None,
            'top': None
        })
        
        # Find the coordinate index for this tile
        coord_idx = None
        for idx, row in coords_df.iterrows():
            if (abs(row['x (mm)'] - tile.stage_x_mm) < 0.001 and 
                abs(row['y (mm)'] - tile.stage_y_mm) < 0.001):
                coord_idx = idx
                break
        
        if coord_idx is not None:
            filename_to_index[tile.filename] = coord_idx
            filenames.append(tile.filename)
    
    # Set up neighbor relationships for the existing function
    if len(alignment_tiles) >= 3:
        center_idx = next(i for i, t in enumerate(alignment_tiles) if t.is_center)
        left_idx = next((i for i, t in enumerate(alignment_tiles) if not t.is_center and 
                        t.grid_row == alignment_tiles[center_idx].grid_row), None)
        
        if left_idx is not None:
            grid_data[center_idx]['left'] = left_idx
    
    grid_df = pd.DataFrame(grid_data)
    
    try:
        # Use the existing function to calculate pixel size
        pixel_size_um = calculate_pixel_size_microns(
            grid=grid_df,
            coords_df=coords_df,
            filename_to_index=filename_to_index,
            filenames=filenames
        )
        print(f"DEBUG: Calculated pixel size using existing function: {pixel_size_um:.4f} um/pixel")
        return pixel_size_um
    except Exception as e:
        print(f"DEBUG: Failed to calculate pixel size using existing function: {e}")
        print("DEBUG: Falling back to default pixel size")
        return 0.325


def detect_acquisition_pattern(coords_df: pd.DataFrame) -> Dict:
    """Detect acquisition pattern from FOV order and stage coordinates.
    
    Returns:
        Dictionary with:
        - pattern: 'row_by_row', 'column_by_column', 'serpentine', or 'sequential'
        - order: list of indices in acquisition order
        - description: human-readable pattern description
    """
    # Sort by FOV number to get acquisition order
    sorted_df = coords_df.sort_values('fov').reset_index(drop=False)
    sorted_df.rename(columns={'index': 'original_idx'}, inplace=True)
    
    # Round stage coordinates to identify rows/columns (with 0.01mm tolerance)
    unique_x = np.round(sorted_df['x (mm)'].unique(), 2)
    unique_y = np.round(sorted_df['y (mm)'].unique(), 2)
    
    print(f"\nDEBUG: Detecting acquisition pattern...")
    print(f"  Total tiles: {len(sorted_df)}")
    print(f"  Unique X positions: {len(unique_x)}")
    print(f"  Unique Y positions: {len(unique_y)}")
    print(f"  FOV range: {sorted_df['fov'].min()} to {sorted_df['fov'].max()}")
    
    # Assign grid row/column based on stage positions
    sorted_df['grid_col'] = sorted_df['x (mm)'].apply(lambda x: np.argmin(np.abs(unique_x - x)))
    sorted_df['grid_row'] = sorted_df['y (mm)'].apply(lambda y: np.argmin(np.abs(unique_y - y)))
    
    # Analyze acquisition pattern
    pattern_type = 'sequential'  # Default: just use FOV order
    description = "Sequential by FOV number"
    
    # Check for row-by-row pattern
    row_changes = (sorted_df['grid_row'].diff() != 0).sum()
    col_changes = (sorted_df['grid_col'].diff() != 0).sum()
    
    print(f"  Row changes: {row_changes}")
    print(f"  Column changes: {col_changes}")
    
    # If most changes are in columns (moving within rows), it's row-by-row
    if col_changes > row_changes * 2:
        pattern_type = 'row_by_row'
        description = "Row-by-row (horizontal scanning)"
        
        # Check for serpentine (alternating directions)
        rows_in_order = sorted_df.groupby('grid_row')['grid_col'].apply(list)
        is_serpentine = False
        
        for i, (row_idx, cols) in enumerate(rows_in_order.items()):
            if len(cols) > 2:
                if i % 2 == 0:  # Even rows should be ascending
                    if cols != sorted(cols):
                        is_serpentine = True
                        break
                else:  # Odd rows should be descending
                    if cols != sorted(cols, reverse=True):
                        is_serpentine = True
                        break
        
        if is_serpentine:
            pattern_type = 'serpentine'
            description = "Serpentine (row-by-row with alternating directions)"
    
    # If most changes are in rows (moving within columns), it's column-by-column
    elif row_changes > col_changes * 2:
        pattern_type = 'column_by_column'
        description = "Column-by-column (vertical scanning)"
    
    print(f"  Detected pattern: {pattern_type}")
    print(f"  Description: {description}")
    
    # Show first few tiles in acquisition order
    print("\n  First 10 tiles in acquisition order:")
    for i in range(min(10, len(sorted_df))):
        row = sorted_df.iloc[i]
        fov_id = row['fov']
        print(f"    {i}: FOV {fov_id} at grid({row['grid_row']},{row['grid_col']}) "
              f"stage({row['x (mm)']:.2f},{row['y (mm)']:.2f})")
    
    return {
        'pattern': pattern_type,
        'order': sorted_df['original_idx'].tolist(),
        'description': description
    }


def apply_fixed_shifts_to_coordinates(coords_df: pd.DataFrame, h_shift_px: float, v_shift_px: float, 
                                     pixel_size_um: float) -> pd.DataFrame:
    """Apply cumulative fixed pixel shifts for unidirectional scanning.
    
    For unidirectional scanning, the shift accumulates within each row:
    - First FOV in row: 0 shift
    - Second FOV in row: 1× shift
    - Third FOV in row: 2× shift
    - First FOV in next row: RESET to 0 shift
    
    This handles systematic stage drift that occurs within each row.
    """
    # TODO: TECHNICAL DEBT - Fixed shift application logic needs to be determined based on 
    # the specific X-Y stage motor behavior and scanning pattern (unidirectional vs bidirectional).
    # Different microscope systems may have different drift patterns that require customized 
    # correction strategies. Current implementation assumes simple row-wise cumulative drift
    # with reset at row boundaries, which may not be appropriate for all systems.
    #
    # Before re-enabling:
    # 1. Characterize actual stage motor drift pattern for the specific system
    # 2. Determine if drift is unidirectional, bidirectional, serpentine, or other
    # 3. Measure drift accumulation behavior (linear, per-row, per-column, etc.)
    # 4. Validate correction approach with test datasets
    
    updated_coords = coords_df.copy()
    
    print(f"\nWARNING: Fixed shift correction is currently disabled due to technical debt.")
    print(f"  The drift pattern must be determined based on X-Y stage motor characteristics.")
    print(f"  No shifts will be applied to coordinates.")
    
    # COMMENTED OUT - Re-enable after characterizing stage motor behavior
    # # Detect acquisition pattern to group tiles by rows
    # pattern_info = detect_acquisition_pattern(coords_df)
    # 
    # print(f"\nDEBUG: Applying fixed shifts for unidirectional scanning")
    # print(f"  Pattern: {pattern_info['description']}")
    # print(f"  Fixed shift per tile within row: h={h_shift_px:.1f}px, v={v_shift_px:.1f}px")
    # 
    # # Convert base pixel shifts to mm
    # h_shift_mm_per_tile = (h_shift_px * pixel_size_um) / 1000.0
    # v_shift_mm_per_tile = (v_shift_px * pixel_size_um) / 1000.0
    # 
    # print(f"  Fixed shift per tile within row: h={h_shift_mm_per_tile:.6f}mm, v={v_shift_mm_per_tile:.6f}mm")
    # 
    # # Sort by FOV to process in acquisition order
    # sorted_df = coords_df.sort_values('fov').reset_index(drop=False)
    # sorted_df.rename(columns={'index': 'original_idx'}, inplace=True)
    # 
    # # Assign grid rows based on Y coordinates
    # unique_y = np.round(sorted_df['y (mm)'].unique(), 2)
    # sorted_df['grid_row'] = sorted_df['y (mm)'].apply(lambda y: np.argmin(np.abs(unique_y - y)))
    # 
    # # Group by grid row and apply cumulative shift within each row
    # current_row = None
    # position_in_row = 0
    # 
    # for idx, row_data in sorted_df.iterrows():
    #     original_idx = row_data['original_idx']
    #     grid_row = row_data['grid_row']
    #     
    #     # Check if we've moved to a new row
    #     if current_row != grid_row:
    #         current_row = grid_row
    #         position_in_row = 0  # RESET shift at start of new row
    #         print(f"\n  Row {grid_row}: Starting new row (shift reset to 0)")
    #     
    #     # Calculate cumulative shift for this position in the row
    #     cumulative_h_shift_mm = h_shift_mm_per_tile * position_in_row
    #     cumulative_v_shift_mm = v_shift_mm_per_tile * position_in_row
    #     
    #     # Apply cumulative shift to this tile
    #     updated_coords.loc[original_idx, 'x (mm)'] += cumulative_h_shift_mm
    #     updated_coords.loc[original_idx, 'y (mm)'] += cumulative_v_shift_mm
    #     
    #     # Debug output for first few tiles in each row
    #     if position_in_row < 3:
    #         print(f"    FOV {row_data['fov']}: position_in_row={position_in_row}, shift=({cumulative_h_shift_mm:.6f}, {cumulative_v_shift_mm:.6f}) mm")
    #     
    #     position_in_row += 1
    # 
    # print(f"\n  Fixed shift correction applied to all rows")
    
    return updated_coords


def launch_fixed_registration(input_directory: str) -> bool:
    """Launch fixed registration dialog and apply shifts to all timepoints."""
    try:
        input_path = Path(input_directory)
        
        # Find all timepoint directories and coordinates files
        timepoint_coords = []
        
        # Check for timepoint structure (0/, 1/, 2/, etc.)
        timepoint_dirs = [d for d in input_path.iterdir() if d.is_dir() and d.name.isdigit()]
        
        if timepoint_dirs:
            # Multi-timepoint structure
            for tp_dir in sorted(timepoint_dirs, key=lambda x: int(x.name)):
                coords_file = tp_dir / "coordinates.csv"
                if coords_file.exists():
                    timepoint_coords.append((tp_dir.name, coords_file))
                    print(f"DEBUG: Found coordinates for timepoint {tp_dir.name}")
        else:
            # Single timepoint structure
            coords_file = input_path / "coordinates.csv"
            if coords_file.exists():
                timepoint_coords.append(("single", coords_file))
            else:
                raise FileNotFoundError("No coordinates.csv files found")
        
        if not timepoint_coords:
            raise FileNotFoundError("No coordinates.csv files found")
        
        # Use first timepoint for alignment dialog
        first_tp_name, first_coords_file = timepoint_coords[0]
        
        # Always start from backup coordinates if available
        original_coords_dir = input_path / "original_coordinates"
        backup_file = original_coords_dir / f"original_coordinates_{first_tp_name}.csv"
        
        if backup_file.exists():
            print(f"DEBUG: REVERTED TO ORIGINAL COORDINATES FILE: {backup_file}")
            print(f"DEBUG: Using backup coordinates from {backup_file}")
            coords_df = read_coordinates_csv(backup_file)
        else:
            print(f"DEBUG: No backup found, using current coordinates from {first_coords_file}")
            coords_df = read_coordinates_csv(first_coords_file)
        
        # Get first alignment triplet efficiently
        image_dir = first_coords_file.parent if first_tp_name != "single" else input_path
        alignment_tiles = select_alignment_tiles(coords_df, image_dir)
        
        # Launch dialog with data for finding more triplets on-demand
        dialog = FixedRegistrationDialog(alignment_tiles, coords_df, image_dir)
        if dialog.exec_() == QDialog.Accepted:
            # Calculate actual pixel size using existing method
            pixel_size_um = calculate_actual_pixel_size(alignment_tiles, coords_df)
            print(f"DEBUG: Using pixel size: {pixel_size_um:.4f} um/pixel")
            print(f"DEBUG: Applying shifts to {len(timepoint_coords)} timepoint(s)")
            
            # Create central backup directory (like tile_registration.py)
            original_coords_dir = input_path / "original_coordinates"
            original_coords_dir.mkdir(exist_ok=True)
            
            # Apply shifts to ALL timepoints using existing backup system
            for tp_name, coords_file in timepoint_coords:
                print(f"DEBUG: Processing timepoint {tp_name}")
                
                # Create backup in central location (like tile_registration.py does)
                backup_path = original_coords_dir / f"original_coordinates_{tp_name}.csv"
                if not backup_path.exists():
                    import shutil
                    shutil.copy2(coords_file, backup_path)
                    print(f"DEBUG: Created backup at {backup_path}")
                else:
                    print(f"DEBUG: Backup already exists at {backup_path}")
                
                # Read coordinates for this timepoint
                tp_coords_df = read_coordinates_csv(coords_file)
                
                # Apply shifts to coordinates
                updated_coords = apply_fixed_shifts_to_coordinates(
                    tp_coords_df, dialog.h_shift_px, dialog.v_shift_px, pixel_size_um
                )
                
                # Save updated coordinates
                updated_coords.to_csv(coords_file, index=False)
                print(f"DEBUG: Updated coordinates saved to {coords_file}")
            
            print(f"DEBUG: Fixed registration applied to all {len(timepoint_coords)} timepoint(s)")
            return True
            
        return False
        
    except Exception as e:
        QMessageBox.critical(None, "Fixed Registration Error", str(e))
        return False
