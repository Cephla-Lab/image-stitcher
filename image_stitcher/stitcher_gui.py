import logging
import sys
import enum
import os
from typing import Any, cast, Union
import pathlib

import napari
import numpy as np
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from PyQt5.QtCore import QThread, Qt, QUrl
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
)

from .parameters import (
    DATETIME_FORMAT,
    OutputFormat,
    ScanPattern,
    StitchingParameters,
    ZLayerSelection,
)
from .stitcher import ProgressCallbacks, Stitcher
from .registration import register_and_update_coordinates
from .registration.tile_registration import process_multiple_timepoints
from .registration._tensor_backend import create_tensor_backend, TensorBackend


# Enum for Flatfield Modes
class FlatfieldModeOption(enum.Enum):
    NONE = "No Flatfield Correction"
    COMPUTE = "Compute Flatfield Correction"
    LOAD = "Load Precomputed Flatfield"

def get_flatfield_mode_from_string(combo_value: str) -> FlatfieldModeOption:
    if combo_value == FlatfieldModeOption.NONE.value:
        return FlatfieldModeOption.NONE
    elif combo_value == FlatfieldModeOption.COMPUTE.value:
        return FlatfieldModeOption.COMPUTE
    elif combo_value == FlatfieldModeOption.LOAD.value:
        return FlatfieldModeOption.LOAD
    else:
        # This case should ideally not be reached if combo box items are populated from the enum.
        raise ValueError(f"Unknown flatfield mode string from combo box: '{combo_value}'")

# TODO(colin): this is almost but not quite the same as the map in
# StitchingComputedParameters.get_channel_color. Reconcile the differences?
CHANNEL_COLORS_MAP = {
    "405": {"hex": 0x3300FF, "name": "blue"},
    "488": {"hex": 0x1FFF00, "name": "green"},
    "561": {"hex": 0xFFCF00, "name": "yellow"},
    "638": {"hex": 0xFF0000, "name": "red"},
    "730": {"hex": 0x770000, "name": "dark red"},
    "R": {"hex": 0xFF0000, "name": "red"},
    "G": {"hex": 0x1FFF00, "name": "green"},
    "B": {"hex": 0x3300FF, "name": "blue"},
}


class DragDropArea(QLabel):
    path_dropped = Signal(str)

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(title, parent)
        self.setMinimumHeight(50)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
        """)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                self.path_dropped.emit(url.toLocalFile())
                self.setText(f"Loaded: {pathlib.Path(url.toLocalFile()).name}")
                self.setStyleSheet("""
                    QLabel {
                        border: 2px solid green;
                        border-radius: 5px;
                        background-color: #e0ffe0;
                    }
                """)
            event.acceptProposedAction()
        else:
            event.ignore()


class StitcherThread(QThread):
    def __init__(self, params: StitchingParameters, perform_registration: bool, image_directory: str, tensor_backend_engine: str | None = None) -> None:
        super().__init__()
        self.params = params
        self.perform_registration = perform_registration
        self.image_directory = image_directory
        self.tensor_backend_engine = tensor_backend_engine
        self.registration_complete = False
        self.registration_error = None
        self.flatfield_corrections: dict[int, np.ndarray] | None = None
        self.flatfield_manifest: pathlib.Path | None = None
        self.callbacks = ProgressCallbacks(
            update_progress=lambda a, b: None,
            getting_flatfields=lambda: None,
            starting_stitching=lambda: None,
            starting_saving=lambda a: None,
            finished_saving=lambda a, b: None,
        )

    def run(self) -> None:
        # Perform registration if requested
        if self.perform_registration:
            try:
                logging.info("Starting registration process...")
                results = process_multiple_timepoints(
                    base_directory=self.image_directory,
                    overlap_diff_threshold=10,    
                    pou=3,                       
                    ncc_threshold=0.5,           
                    edge_width=195,
                    tensor_backend_engine=self.tensor_backend_engine,
                    flatfield_corrections=self.flatfield_corrections
                )
                logging.info("Registration completed successfully")
                self.registration_complete = True
            except Exception as e:
                logging.error(f"Registration failed: {e}")
                self.registration_error = str(e)
                return  # Don't proceed with stitching if registration failed
        
        # Only run stitching if registration was successful or not requested
        if not self.perform_registration or self.registration_complete:
            logging.info("Starting stitching process...")
            # Create a new Stitcher instance to ensure it uses the updated coordinates
            stitcher = Stitcher(
                self.params,
                self.callbacks
            )
            stitcher.run()
        else:
            logging.error("Stitching skipped due to registration failure")

    error = Signal(str)
    finished = Signal()


class RegistrationThread(QThread):
    def __init__(
        self, 
        image_directory: str, 
        csv_path: str | None, 
        output_csv_path: str | None, 
        tensor_backend_engine: str | None = None,
        flatfield_corrections: dict[int, np.ndarray] | None = None,
        flatfield_manifest: pathlib.Path | None = None
    ) -> None:
        super().__init__()
        self.image_directory = image_directory
        self.csv_path = csv_path
        self.output_csv_path = output_csv_path
        self.tensor_backend_engine = tensor_backend_engine
        self.flatfield_corrections = flatfield_corrections
        self.flatfield_manifest = flatfield_manifest

    def run(self) -> None:
        try:
            # Process all timepoints
            results = process_multiple_timepoints(
                base_directory=self.image_directory,
                overlap_diff_threshold=10,    
                pou=3,                      
                ncc_threshold=0.5,           
                edge_width=195,
                tensor_backend_engine=self.tensor_backend_engine,
                flatfield_corrections=self.flatfield_corrections
            )
            
            # Emit success signal with number of processed timepoints
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))

    error = Signal(str)
    finished = Signal()


class StitchingGUI(QWidget):
    # Signals for progress indicators. QT dictates these must be defined at the class level.
    update_progress = Signal(int, int)
    getting_flatfields = Signal()
    starting_stitching = Signal()
    starting_saving = Signal(bool)
    finished_saving = Signal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self.stitcher: StitcherThread | None = (
            None  # Stitcher is initialized when needed
        )
        self.inputDirectory: str | None = (
            None  # This will be set by the directory selection
        )
        self.output_path = ""
        self.dtype: np.dtype | None = None
        self.flatfield_manifest: pathlib.Path | None = None
        self.computed_flatfields: dict[int, np.ndarray] | None = None  # Store computed flatfields
        self.flatfield_computed_params: Any | None = None  # Store computed parameters used for flatfield
        self.initUI()

    def initUI(self) -> None:
        self.mainLayout = QGridLayout(self) # Main layout for the window
        self.setLayout(self.mainLayout)
        self.mainLayout.setSpacing(10)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)

        # Input directory section
        self.inputDirLabel = QLabel("Acquisition Directory:", self)
        self.mainLayout.addWidget(self.inputDirLabel, 0, 0)
        self.inputDirDropArea = DragDropArea("Drag & Drop Input Directory Here", self)
        self.inputDirDropArea.path_dropped.connect(self.onInputDirectoryDropped)
        self.mainLayout.addWidget(self.inputDirDropArea, 0, 1)

        # Start row counter
        row = 1
        
        # Compute Backend section
        self.tensorBackendLabel = QLabel("Compute Backend:", self)
        self.mainLayout.addWidget(self.tensorBackendLabel, row, 0)
        self.tensorBackendCombo = QComboBox(self)
        self.mainLayout.addWidget(self.tensorBackendCombo, row, 1)
        row += 1
        
        # Initialize backend options
        self._populate_tensor_backend_options()
        
        # Flatfield Correction section (moved before registration)
        self.flatfieldModeLabel = QLabel("Flatfield Correction:", self)
        self.mainLayout.addWidget(self.flatfieldModeLabel, row, 0)
        self.flatfieldModeCombo = QComboBox(self)
        self.flatfieldModeCombo.addItems(
            [
                "No Flatfield Correction",
                "Compute Flatfield Correction",
                "Load Precomputed Flatfield",
            ]
        )
        self.flatfieldModeCombo.currentIndexChanged.connect(self.onFlatfieldModeChanged)
        self.mainLayout.addWidget(self.flatfieldModeCombo, row, 1)
        row += 1

        self.flatfieldLoadLabel = QLabel("Load Flatfield:", self)
        self.mainLayout.addWidget(self.flatfieldLoadLabel, row, 0)
        self.flatfieldLoadLabel.setVisible(False)
        self.loadFlatfieldDropArea = DragDropArea("Drag & Drop Flatfield Directory Here", self)
        self.loadFlatfieldDropArea.path_dropped.connect(self.onLoadFlatfieldDropped)
        self.mainLayout.addWidget(self.loadFlatfieldDropArea, row, 1)
        self.loadFlatfieldDropArea.setVisible(False)
        row += 1
        
        # Registration checkbox - when checked, registration runs before stitching
        self.runRegistrationLabel = QLabel("Run Registration:", self)
        self.mainLayout.addWidget(self.runRegistrationLabel, row, 0)
        self.runRegistrationCheckbox = QCheckBox(self)
        self.runRegistrationCheckbox.setChecked(False)  # Default: unchecked
        self.mainLayout.addWidget(self.runRegistrationCheckbox, row, 1)
        row += 1

        # Output format section
        self.outputFormatLabel = QLabel("Output Format:", self)
        self.mainLayout.addWidget(self.outputFormatLabel, row, 0)
        self.outputFormatCombo = QComboBox(self)
        self.outputFormatCombo.addItems(["OME-ZARR"])  # Removed OME-TIFF as requested
        self.mainLayout.addWidget(self.outputFormatCombo, row, 1)
        row += 1

        # Output compression section
        self.compressionLabel = QLabel("Output Compression:", self)
        self.mainLayout.addWidget(self.compressionLabel, row, 0)
        self.outputCompression = QComboBox(self)
        self.outputCompression.addItems(["default", "none"])
        self.mainLayout.addWidget(self.outputCompression, row, 1)
        row += 1

        # Pyramid checkbox section - now with checkbox on the right
        self.pyramidLabel = QLabel("Infer levels for output image pyramid:", self)
        self.mainLayout.addWidget(self.pyramidLabel, row, 0)
        self.pyramidCheckbox = QCheckBox(self)
        self.pyramidCheckbox.setChecked(True)
        self.mainLayout.addWidget(self.pyramidCheckbox, row, 1)
        row += 1

        # Z-Stack Options section
        self.zLayerLabel = QLabel("Processing Mode:", self)
        self.mainLayout.addWidget(self.zLayerLabel, row, 0)
        self.zLayerModeCombo = QComboBox(self)
        self.zLayerModeCombo.addItems(["Middle Layer", "All Layers", "Specific Layer", "Maximum Intensity Projection (MIP)"])
        self.zLayerModeCombo.currentIndexChanged.connect(self.onZLayerModeChanged)
        self.mainLayout.addWidget(self.zLayerModeCombo, row, 1)
        row += 1

        self.zLayerSpinLabel = QLabel("Select Z-Layer Index:", self)
        self.mainLayout.addWidget(self.zLayerSpinLabel, row, 0)
        self.zLayerSpinLabel.setVisible(False) 

        self.zLayerSpinBox = QSpinBox(self)
        self.zLayerSpinBox.setMinimum(0)
        self.zLayerSpinBox.setMaximum(999)  # Will be updated based on actual data
        self.mainLayout.addWidget(self.zLayerSpinBox, row, 1)
        self.zLayerSpinBox.setVisible(False) 
        row += 1

        # Status and Progress section
        self.statusLabel = QLabel("Status: Ready", self)
        self.mainLayout.addWidget(self.statusLabel, row, 0, 1, 2)
        row += 1

        self.progressBar = QProgressBar(self)
        self.progressBar.hide()
        self.mainLayout.addWidget(self.progressBar, row, 0, 1, 2)
        row += 1

        # Action Buttons
        self.startBtn = QPushButton("Start Stitching", self)
        self.startBtn.clicked.connect(self.onStitchingStart)
        self.startBtn.setEnabled(False)  # Disabled until input directory is selected
        self.mainLayout.addWidget(self.startBtn, row, 0, 1, 2)  # Span both columns
        row += 1

        # Output file path section - create horizontal layout for path + button
        outputPathLayout = QHBoxLayout()
        self.outputPathEdit = QLineEdit(self)
        self.outputPathEdit.setPlaceholderText("Path to stitched output file...")
        self.outputPathEdit.setReadOnly(False)  # Allow manual editing
        outputPathLayout.addWidget(self.outputPathEdit)
        
        self.browseOutputBtn = QPushButton("Browse Files", self)
        self.browseOutputBtn.clicked.connect(self.onBrowseOutputFile)
        self.browseOutputBtn.setMaximumWidth(120)  # Make button smaller
        outputPathLayout.addWidget(self.browseOutputBtn)
        
        self.mainLayout.addLayout(outputPathLayout, row, 0, 1, 2)  # Span both columns
        row += 1

        self.viewBtn = QPushButton("View Output", self)
        self.viewBtn.clicked.connect(self.onViewOutput)
        self.viewBtn.setEnabled(False)
        self.mainLayout.addWidget(self.viewBtn, row, 0, 1, 2)  # Span both columns
        row += 1
        
        # Add stretch to push everything to the top
        self.mainLayout.setRowStretch(row, 1) 

        self.setWindowTitle("Cephla Image Stitcher")
        self.setGeometry(300, 300, 600, 400)
        self.show()

    def onInputDirectoryDropped(self, path: str) -> None:
        path_obj = pathlib.Path(path)
        # Detect acquisition directory: either the dropped directory itself contains the
        # acquisition parameters file, or its parent does (if the user dropped a specific
        # time-point folder such as "0").
        if (path_obj / "acquisition parameters.json").exists():
            acquisition_dir = path_obj
        elif (path_obj.name.isdigit() and (path_obj.parent / "acquisition parameters.json").exists()):
            acquisition_dir = path_obj.parent
        else:
            QMessageBox.warning(
                self,
                "Input Error",
                "Selected directory does not look like a Cephla acquisition folder.",
            )
            return

        self.inputDirectory = str(acquisition_dir)
        
        # Enable stitching button
        self.startBtn.setEnabled(True)
        
        # Detect and display the acquisition format
        try:
            temp_params = StitchingParameters(
                input_folder=self.inputDirectory,
                output_format=OutputFormat.ome_zarr, 
                scan_pattern=ScanPattern.unidirectional,
            )
            temp_stitcher = Stitcher(temp_params)
            
            # Check if it's multi-page TIFF format
            if temp_stitcher.computed_parameters.is_multipage_tiff_format():
                format_info = " (Multi-page TIFF)"
            else:
                format_info = " (Individual files)"
                
            # Visually mark the drop area
            self.inputDirDropArea.setText(f"Loaded: {acquisition_dir.name}{format_info}")
            self.inputDirDropArea.setStyleSheet("""
                QLabel {border: 2px solid green; border-radius: 5px; background-color: #e0ffe0;}
            """)
            self.probeDatasetForZLayers()
            
        except Exception as e:
            logging.warning(f"Could not detect acquisition format: {e}")
            # Fallback to original behavior
            self.inputDirDropArea.setText(f"Loaded: {acquisition_dir.name}")
            self.inputDirDropArea.setStyleSheet("""
                QLabel {border: 2px solid green; border-radius: 5px; background-color: #e0ffe0;}
            """)
            self.probeDatasetForZLayers()

    def selectInputDirectory(self) -> None: # Kept for now, can be removed if button is fully replaced
        dir = QFileDialog.getExistingDirectory(self, "Select Input Image Folder")
        if dir:
            self.inputDirectory = dir
            
            # Detect and display the acquisition format
            try:
                temp_params = StitchingParameters(
                    input_folder=self.inputDirectory,
                    output_format=OutputFormat.ome_zarr, 
                    scan_pattern=ScanPattern.unidirectional,
                )
                temp_stitcher = Stitcher(temp_params)
                
                # Check if it's multi-page TIFF format
                if temp_stitcher.computed_parameters.is_multipage_tiff_format():
                    format_info = " (Multi-page TIFF)"
                else:
                    format_info = " (Individual files)"
                    
                self.inputDirDropArea.setText(f"Loaded: {pathlib.Path(dir).name}{format_info}")
                self.inputDirDropArea.setStyleSheet("""QLabel {border: 2px solid green; border-radius: 5px; background-color: #e0ffe0;}""")
                self.probeDatasetForZLayers()
                
            except Exception as e:
                logging.warning(f"Could not detect acquisition format: {e}")
                # Fallback to original behavior
                self.inputDirDropArea.setText(f"Loaded: {pathlib.Path(dir).name}")
                self.inputDirDropArea.setStyleSheet("""QLabel {border: 2px solid green; border-radius: 5px; background-color: #e0ffe0;}""")
                self.probeDatasetForZLayers()

    def probeDatasetForZLayers(self) -> None:
        if not self.inputDirectory:
            return
        try:
            temp_params = StitchingParameters(
                input_folder=self.inputDirectory,
                output_format=OutputFormat.ome_zarr, 
                scan_pattern=ScanPattern.unidirectional,
            )
            temp_stitcher = Stitcher(temp_params)
            num_z = temp_stitcher.computed_parameters.num_z

            self.zLayerSpinBox.setMaximum(num_z - 1)
            self.zLayerSpinLabel.setText(f"Select Z-Layer Index (0-{num_z - 1}):")

            if self.zLayerModeCombo.currentIndex() == 0:  # Middle Layer
                middle_idx = num_z // 2
                self.zLayerLabel.setText(
                    f"Processing Mode (total layers: {num_z}, middle: {middle_idx}):"
                )
            else:
                self.zLayerLabel.setText(
                    f"Processing Mode (total layers: {num_z}):"
                )

        except Exception as e:
            logging.warning(f"Could not probe dataset for z-layers: {e}")
            self.zLayerLabel.setText("Processing Mode:")

    def onStitchingStart(self) -> None:
        """Start stitching from GUI (registration is separate)."""
        if not self.inputDirectory:
            QMessageBox.warning(
                self, "Input Error", "Please select an input directory."
            )
            return

        try:
            # Get flatfield mode selection
            flatfield_mode = get_flatfield_mode_from_string(self.flatfieldModeCombo.currentText())
            
            # Check if registration should be run first based on checkbox state
            perform_registration = self.runRegistrationCheckbox.isChecked()
            
            # Handle flatfield computation/loading if needed for registration
            flatfield_corrections = None
            flatfield_manifest = None
            
            if perform_registration and flatfield_mode != FlatfieldModeOption.NONE:
                if flatfield_mode == FlatfieldModeOption.COMPUTE:
                    # Check if flatfields already exist
                    acquisition_folder = pathlib.Path(self.inputDirectory)
                    flatfield_dir = acquisition_folder / "flatfields"
                    auto_flatfield_manifest = flatfield_dir / "flatfield_manifest.json"
                    
                    if auto_flatfield_manifest.exists():
                        # Try to load existing flatfields instead of recomputing
                        self.statusLabel.setText("Status: Loading existing flatfields...")
                        QApplication.processEvents()
                        logging.info(f"Found existing flatfields at {auto_flatfield_manifest}, loading instead of recomputing")
                        
                        try:
                            from .flatfield_utils import load_flatfield_correction
                            temp_params = StitchingParameters(
                                input_folder=self.inputDirectory,
                                output_format=OutputFormat.ome_zarr,
                                scan_pattern=ScanPattern.unidirectional
                            )
                            temp_stitcher = Stitcher(temp_params)
                            
                            flatfield_corrections = load_flatfield_correction(
                                auto_flatfield_manifest,
                                temp_stitcher.computed_parameters
                            )
                            
                            if flatfield_corrections:
                                self.computed_flatfields = flatfield_corrections
                                self.flatfield_computed_params = temp_stitcher.computed_parameters
                                flatfield_manifest = auto_flatfield_manifest
                                logging.info(f"Successfully loaded existing flatfields from {auto_flatfield_manifest}")
                            else:
                                # Manifest exists but no valid flatfields, compute new ones
                                logging.warning("Existing flatfield manifest found but no valid flatfields loaded. Computing new ones.")
                                flatfield_corrections = None
                                
                        except Exception as e:
                            logging.warning(f"Failed to load existing flatfields: {e}. Computing new ones.")
                            flatfield_corrections = None
                    else:
                        flatfield_corrections = None
                    
                    # Compute flatfields if not loaded from existing
                    if not flatfield_corrections:
                        self.statusLabel.setText("Status: Computing flatfields...")
                        self.progressBar.setRange(0, 0)
                        self.progressBar.show()
                        QApplication.processEvents()  # Update UI
                        
                        try:
                            # Create temporary stitching parameters to compute flatfields
                            temp_params = StitchingParameters(
                                input_folder=self.inputDirectory,
                                output_format=OutputFormat.ome_zarr,
                                scan_pattern=ScanPattern.unidirectional,
                                apply_flatfield=False  # Don't apply, just compute
                            )
                            temp_stitcher = Stitcher(temp_params)
                            
                            # Compute flatfields
                            from .flatfield_correction import compute_flatfield_correction
                            flatfield_corrections = compute_flatfield_correction(
                                temp_stitcher.computed_parameters,
                                lambda: None  # No progress callback needed
                            )
                            
                            # Save flatfields for later use
                            if flatfield_corrections:
                                from .flatfield_utils import save_flatfield_correction
                                flatfield_manifest = save_flatfield_correction(
                                    flatfield_corrections,
                                    temp_stitcher.computed_parameters,
                                    flatfield_dir
                                )
                                self.computed_flatfields = flatfield_corrections
                                self.flatfield_computed_params = temp_stitcher.computed_parameters
                                logging.info(f"Computed flatfields saved to {flatfield_dir}, manifest at {flatfield_manifest}")
                            
                        except Exception as e:
                            logging.error(f"Failed to compute flatfields: {e}")
                            QMessageBox.warning(
                                self, "Flatfield Warning", 
                                f"Failed to compute flatfields: {e}\nContinuing registration without flatfield correction."
                            )
                            flatfield_corrections = None
                        
                elif flatfield_mode == FlatfieldModeOption.LOAD:
                    # Load precomputed flatfields
                    if self.flatfield_manifest:
                        flatfield_manifest = self.flatfield_manifest
                        try:
                            from .flatfield_utils import load_flatfield_correction
                            temp_params = StitchingParameters(
                                input_folder=self.inputDirectory,
                                output_format=OutputFormat.ome_zarr,
                                scan_pattern=ScanPattern.unidirectional
                            )
                            temp_stitcher = Stitcher(temp_params)
                            
                            flatfield_corrections = load_flatfield_correction(
                                self.flatfield_manifest,
                                temp_stitcher.computed_parameters
                            )
                            self.computed_flatfields = flatfield_corrections
                            logging.info("Loaded flatfield corrections for registration")
                        except Exception as e:
                            logging.error(f"Failed to load flatfields: {e}")
                            QMessageBox.warning(
                                self, "Flatfield Warning",
                                f"Failed to load flatfields: {e}\nContinuing registration without flatfield correction."
                            )
                            flatfield_corrections = None
                            flatfield_manifest = None
                    else:
                        QMessageBox.warning(
                            self, "Flatfield Warning",
                            "No flatfield directory selected.\nContinuing registration without flatfield correction."
                        )
            
            # Create parameters from UI state
            format_text = self.outputFormatCombo.currentText()
            if format_text == "OME-ZARR":
                output_format = OutputFormat.ome_zarr
            else:
                QMessageBox.critical(self, "Internal Error", f"Invalid output format selected: {format_text}")
                return

            apply_flatfield = flatfield_mode != FlatfieldModeOption.NONE
            # For stitching: use explicit manifest only when LOAD mode
            # For COMPUTE mode, let stitcher auto-discover the saved flatfields
            if flatfield_mode == FlatfieldModeOption.LOAD:
                flatfield_manifest = self.flatfield_manifest
            else:
                flatfield_manifest = None

            # Determine z-layer selection strategy
            z_layer_mode = self.zLayerModeCombo.currentIndex()
            if z_layer_mode == 0:  # Middle Layer
                z_layer_selection = "middle"
            elif z_layer_mode == 1:  # All Layers
                z_layer_selection = "all"
            elif z_layer_mode == 2:  # Specific Layer
                z_layer_selection = str(self.zLayerSpinBox.value())
            else:  # MIP
                z_layer_selection = "all"  # Use "all" for MIP since we need all layers

            params = StitchingParameters(
                input_folder=self.inputDirectory,
                output_format=output_format,
                scan_pattern=ScanPattern.unidirectional,
                apply_flatfield=apply_flatfield,
                flatfield_manifest=flatfield_manifest,
                z_layer_selection=z_layer_selection,
                apply_mip=(z_layer_mode == 3),  # Set apply_mip based on the combo box index
            )
            
            # Get selected backend if registration is requested
            tensor_backend_engine = None
            if perform_registration:
                tensor_backend_engine = self._get_selected_tensor_backend()
            
            self.stitcher = StitcherThread(
                params=params,
                perform_registration=perform_registration,
                image_directory=self.inputDirectory,
                tensor_backend_engine=tensor_backend_engine
            )
            
            # Pass flatfield corrections to the stitcher thread for registration
            if perform_registration and flatfield_corrections:
                self.stitcher.flatfield_corrections = flatfield_corrections
                self.stitcher.flatfield_manifest = flatfield_manifest

            # Set up callbacks
            self.stitcher.callbacks = ProgressCallbacks(
                update_progress=self.update_progress.emit,
                getting_flatfields=self.getting_flatfields.emit,
                starting_stitching=self.starting_stitching.emit,
                starting_saving=self.starting_saving.emit,
                finished_saving=self.finished_saving.emit,
            )

            # Connect signals
            self.stitcher.error.connect(self.onStitchingError)
            self.stitcher.finished.connect(self.onStitchingFinished)
            self.setupConnections()

            # Start processing
            self.statusLabel.setText("Status: Stitching...")
            self.stitcher.start()
            self.progressBar.show()

        except Exception as e:
            QMessageBox.critical(self, "Stitching Error", str(e))
            self.statusLabel.setText("Status: Error Encountered")

    def onStitchingError(self, error_msg: str) -> None:
        """Handle stitching errors."""
        QMessageBox.critical(self, "Stitching Error", error_msg)
        self.statusLabel.setText("Status: Error Encountered")
        self.progressBar.hide()

    def onStitchingFinished(self) -> None:
        """Handle completion of stitching."""
        self.statusLabel.setText("Status: Stitching completed")
        self.progressBar.hide()
        
        # Update registration status if it was performed
        if self.stitcher and self.stitcher.perform_registration:
            if self.stitcher.registration_complete:
                self.statusLabel.setText("Status: Registration complete")
            elif self.stitcher.registration_error:
                self.statusLabel.setText(f"Status: Registration failed - {self.stitcher.registration_error}")
        
        logging.info("Image stitching has been completed successfully.")

    def onFlatfieldModeChanged(self, idx: int) -> None:
        show_load_option = (idx == 2)
        self.flatfieldLoadLabel.setVisible(show_load_option)
        self.loadFlatfieldDropArea.setVisible(show_load_option)
        if not show_load_option:
            self.flatfield_manifest = None
            self.loadFlatfieldDropArea.setText("Drag & Drop Flatfield Directory Here")
            self.loadFlatfieldDropArea.setStyleSheet("""
                QLabel {
                    border: 2px dashed #aaa;
                    border-radius: 5px;
                    background-color: #f0f0f0;
                }
            """)
        elif show_load_option and self.flatfield_manifest:
             self.loadFlatfieldDropArea.setText(f"Loaded: {self.flatfield_manifest.name}")
             self.loadFlatfieldDropArea.setStyleSheet("""
                QLabel {
                    border: 2px solid green;
                    border-radius: 5px;
                    background-color: #e0ffe0;
                }
            """)

    def onLoadFlatfieldDropped(self, path: str) -> None:
        path_obj = pathlib.Path(path)
        if path_obj.is_dir():
            # Look for flatfield_manifest.json in the directory
            manifest_path = path_obj / "flatfield_manifest.json"
            if manifest_path.exists():
                self.flatfield_manifest = manifest_path
                self.loadFlatfieldDropArea.setText(f"Loaded: {manifest_path.name}")
                self.loadFlatfieldDropArea.setStyleSheet("""
                    QLabel {
                        border: 2px solid green;
                        border-radius: 5px;
                        background-color: #e0ffe0;
                    }
                """)
            else:
                QMessageBox.warning(self, "Input Error", "No flatfield_manifest.json found in the dropped directory.")
                self.flatfield_manifest = None
                self.loadFlatfieldDropArea.setText("Drag & Drop Flatfield Directory Here")
                self.loadFlatfieldDropArea.setStyleSheet("""
                    QLabel {
                        border: 2px dashed #aaa;
                        border-radius: 5px;
                        background-color: #f0f0f0;
                    }
                """)
        else:
            QMessageBox.warning(self, "Input Error", "Please drop a directory for flatfield data.")
            self.flatfield_manifest = None
            self.loadFlatfieldDropArea.setText("Drag & Drop Flatfield Directory Here")
            self.loadFlatfieldDropArea.setStyleSheet("""
                QLabel {
                    border: 2px dashed #aaa;
                    border-radius: 5px;
                    background-color: #f0f0f0;
                }
            """)

    def onLoadFlatfield(self) -> None: # Kept for now, can be removed if button is fully replaced
        directory = QFileDialog.getExistingDirectory(self, "Select Flatfield Folder")
        if directory:
            path_obj = pathlib.Path(directory)
            manifest_path = path_obj / "flatfield_manifest.json"
            if manifest_path.exists():
                self.flatfield_manifest = manifest_path
                self.loadFlatfieldDropArea.setText(f"Loaded: {manifest_path.name}")
                self.loadFlatfieldDropArea.setStyleSheet("""
                    QLabel {
                        border: 2px solid green;
                        border-radius: 5px;
                        background-color: #e0ffe0;
                    }
                """)
            else:
                QMessageBox.warning(self, "Input Error", "No flatfield_manifest.json found in the selected directory.")
                self.flatfield_manifest = None
                self.loadFlatfieldDropArea.setText("Drag & Drop Flatfield Directory Here")
                self.loadFlatfieldDropArea.setStyleSheet("""
                    QLabel {
                        border: 2px dashed #aaa;
                        border-radius: 5px;
                        background-color: #f0f0f0;
                    }
                """)
        else:
            self.flatfield_manifest = None
            self.loadFlatfieldDropArea.setText("Drag & Drop Flatfield Directory Here")
            self.loadFlatfieldDropArea.setStyleSheet("""
                QLabel {
                    border: 2px dashed #aaa;
                    border-radius: 5px;
                    background-color: #f0f0f0;
                }
            """)

    def onZLayerModeChanged(self, idx: int) -> None:
        """Handle z-layer mode selection changes."""
        # Show/hide specific layer controls based on selection
        if idx == 2:  # "Specific Layer" selected
            self.zLayerSpinLabel.setVisible(True)
            self.zLayerSpinBox.setVisible(True)
        else:
            self.zLayerSpinLabel.setVisible(False)
            self.zLayerSpinBox.setVisible(False)

    def _populate_tensor_backend_options(self) -> None:
        """Populate the tensor backend dropdown with available options for registration."""
        # Define backend options with display names
        backend_options = [
            ("Auto (Recommended)", None),
            ("NumPy (CPU)", "numpy"),
            ("PyTorch (GPU/CPU)", "torch"), 
            ("CuPy (NVIDIA GPU)", "cupy")
        ]
        
        # Test which backends are actually available
        self._available_backends = []
        for display_name, engine_name in backend_options:
            if engine_name is None:  # Auto option
                self._available_backends.append((display_name, engine_name))
            else:
                try:
                    # Test if backend can be created and actually works (no fallback)
                    backend = create_tensor_backend(engine_name, allow_fallback=False)
                    gpu_status = " (GPU)" if backend.is_gpu else " (CPU)"
                    display_with_status = display_name.replace("(GPU/CPU)", gpu_status).replace("(NVIDIA GPU)", gpu_status)
                    self._available_backends.append((display_with_status, engine_name))
                except Exception:
                    # Backend not available, skip it
                    continue
        
        # Populate the backend dropdown
        self._backend_mapping = {}
        for i, (display_name, engine_name) in enumerate(self._available_backends):
            self.tensorBackendCombo.addItem(display_name)
            self._backend_mapping[i] = engine_name
        
        # Connect change handler
        self.tensorBackendCombo.currentIndexChanged.connect(self.onTensorBackendChanged)
        
        # Set default to auto
        if len(self._available_backends) > 0:
            self.tensorBackendCombo.setCurrentIndex(0)
            self._update_backend_status()

    def onTensorBackendChanged(self, idx: int) -> None:
        """Handle tensor backend selection changes."""
        self._update_backend_status()

    def _update_backend_status(self) -> None:
        """Update status to show current backend selection."""
        try:
            selected_idx = self.tensorBackendCombo.currentIndex()
            engine_name = self._backend_mapping.get(selected_idx)
            
            if engine_name is None:
                # Auto selection - detect current backend
                current_backend = create_tensor_backend(None)
                backend_name = current_backend.name
                gpu_status = "GPU" if current_backend.is_gpu else "CPU"
                status_text = f"Auto-selected: {backend_name.upper()} ({gpu_status})"
            else:
                # Manual selection
                try:
                    selected_backend = create_tensor_backend(engine_name, allow_fallback=False)
                    backend_name = selected_backend.name
                    gpu_status = "GPU" if selected_backend.is_gpu else "CPU"
                    status_text = f"Selected: {backend_name.upper()} ({gpu_status})"
                except Exception as e:
                    status_text = f"Backend {engine_name} unavailable: {str(e)[:50]}..."
            
            # Update tooltip with backend info
            self.tensorBackendCombo.setToolTip(status_text)
            
        except Exception as e:
            self.tensorBackendCombo.setToolTip(f"Backend error: {str(e)[:50]}...")

    def _get_selected_tensor_backend(self) -> str | None:
        """Get the currently selected tensor backend engine name."""
        selected_idx = self.tensorBackendCombo.currentIndex()
        return self._backend_mapping.get(selected_idx)

    def setupConnections(self) -> None:
        assert self.stitcher is not None
        self.update_progress.connect(self.updateProgressBar)
        self.getting_flatfields.connect(
            lambda: self.statusLabel.setText("Status: Calculating Flatfields...")
        )
        self.starting_stitching.connect(
            lambda: self.statusLabel.setText("Status: Stitching FOVS...")
        )
        self.starting_saving.connect(self.onStartingSaving)
        self.finished_saving.connect(self.onFinishedSaving)

    def updateProgressBar(self, value: int, maximum: int) -> None:
        self.progressBar.setRange(0, maximum)
        self.progressBar.setValue(value)

    def onStartingSaving(self, stitch_complete: bool = False) -> None:
        if stitch_complete:
            self.statusLabel.setText("Status: Saving Complete Acquisition Image...")
        else:
            self.statusLabel.setText("Status: Saving Stitched Image...")
        self.progressBar.setRange(0, 0)  # Indeterminate mode
        self.progressBar.show()
        self.statusLabel.show()

    def onFinishedSaving(self, path: str, dtype: Any) -> None:
        self.progressBar.setValue(0)
        self.progressBar.hide()
        self.viewBtn.setEnabled(True)
        self.statusLabel.setText("Saving Completed. Ready to View.")
        self.output_path = path
        self.outputPathEdit.setText(path)  # Update the output path field
        self.dtype = np.dtype(dtype)
        if dtype == np.uint16:
            c = [0, 65535]
        elif dtype == np.uint8:
            c = [0, 255]
        else:
            c = None
        self.contrast_limits = c
        self.setGeometry(300, 300, 500, 200)

    def onErrorOccurred(self, error: Any) -> None:
        QMessageBox.critical(self, "Error", f"Error while processing: {error}")
        self.statusLabel.setText("Error Occurred!")

    def onBrowseOutputFile(self) -> None:
        """Handle the 'Browse Files' button click to select a stitched output file."""
        # Allow selecting both directories (for .ome.zarr) and files (for .ome.tiff)
        options = QFileDialog.Options()
        
        # Start with directory selection dialog for .ome.zarr files
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Stitched Output Directory (.ome.zarr)",
            self.inputDirectory if self.inputDirectory else "",
            options=options
        )
        
        if selected_dir:
            # Validate that this is a stitched output
            selected_path = pathlib.Path(selected_dir)
            if selected_path.name.endswith('.ome.zarr') or selected_path.name.endswith('_stitched'):
                self.outputPathEdit.setText(selected_dir)
                self.output_path = selected_dir
                self.viewBtn.setEnabled(True)
            else:
                QMessageBox.warning(
                    self, 
                    "Invalid Selection", 
                    "Please select a valid stitched output directory (.ome.zarr)"
                )
        # Note: Could also add support for .ome.tiff file selection if needed

    def onViewOutput(self) -> None:
        """Handle the 'View Output' button click.
        
        This method intelligently determines whether to launch the HCS Grid Viewer
        (for wellplate datasets with multiple timepoints) or open directly in Napari
        (for single files or single timepoint datasets).
        """
        # Configure logging to see debug output
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Get path from the text field (allows user to browse or manually edit)
        output_path = self.outputPathEdit.text().strip()
        logging.info(f"onViewOutput called with output_path: {output_path}")
        
        if not output_path:
            QMessageBox.warning(self, "View Error", "No output path set. Please browse to select a file or complete stitching first.")
            return
        
        # Validate path exists
        if not pathlib.Path(output_path).exists():
            QMessageBox.warning(self, "View Error", f"The specified path does not exist:\n{output_path}")
            return
        
        # Check if this is a wellplate dataset using the same logic as HCS viewer
        dataset_dir = self._get_dataset_directory_from_output_path(output_path)
        logging.info(f"Extracted dataset directory: {dataset_dir}")
        
        is_wellplate = self._detect_wellplate_dataset(dataset_dir)
        logging.info(f"Wellplate detection result: {is_wellplate}")
        
        if is_wellplate:
            # This is a wellplate dataset - use HCS viewer
            logging.info(f"Launching HCS viewer with dataset path: {dataset_dir}")
            try:
                self._launch_hcs_viewer(dataset_dir)
            except Exception as e:
                logging.error(f"An error occurred while launching HCS viewer: {e}")
                # Fall back to direct Napari opening
                logging.info("Falling back to direct Napari opening.")
                self._open_in_napari(output_path)
        else:
            # Single file or single timepoint - open directly in Napari
            logging.info(f"Opening in Napari with path: {output_path}")
            self._open_in_napari(output_path)
    
    def _get_dataset_directory_from_output_path(self, output_path: str) -> pathlib.Path:
        """Extract the dataset directory from the output path using the same logic as HCS viewer"""
        output_dir = pathlib.Path(output_path)
        logging.info(f"Processing output_path: {output_path}")
        logging.info(f"Initial output_dir: {output_dir}")
        logging.info(f"output_dir.name: {output_dir.name}")
        logging.info(f"output_dir.is_dir(): {output_dir.is_dir()}")
        
        # The output_path points to a specific file/directory like:
        # /path/to/output/2_stitched/RegionA_stitched.ome.zarr
        # We need to go up to the parent directory that contains all timepoint dirs
        if output_dir.name.endswith('.ome.zarr') or output_dir.name.endswith('.ome.tiff'):
            # This is a zarr directory or tiff file, go up two levels: file -> timepoint_dir -> output_folder
            output_dir = output_dir.parent.parent
            logging.info(f"Path ends with .ome.zarr/.ome.tiff, going up 2 levels to: {output_dir}")
        elif output_dir.is_dir() and output_dir.name.endswith("_stitched"):
            # If we got a timepoint directory, go up one level to the output folder
            output_dir = output_dir.parent
            logging.info(f"Path is timepoint directory, going up 1 level to: {output_dir}")
        else:
            logging.info(f"No path adjustment needed, keeping: {output_dir}")
        
        logging.info(f"Final dataset directory: {output_dir}")
        return output_dir
    
    def _detect_wellplate_dataset(self, directory: pathlib.Path) -> bool:
        """Detect wellplate datasets using the same logic as HCS viewer"""
        try:
            import re
            
            # Check for timepoint directories with strict pattern: digit(s)_stitched
            timepoint_dirs = []
            for item in directory.iterdir():
                if (item.is_dir() and 
                    re.match(r'^\d+_stitched$', item.name)):  # Only match numeric timepoint dirs
                    timepoint_dirs.append(item)
            
            logging.info(f"Checking dataset directory: {directory}")
            logging.info(f"Found {len(timepoint_dirs)} timepoint directories: {[d.name for d in timepoint_dirs]}")
            
            if not timepoint_dirs:
                logging.info("No timepoint directories found - not a wellplate dataset")
                return False
            
            # Check for OME-ZARR files in timepoint directories and analyze region names
            total_zarr_files = 0
            wellplate_regions = set()
            
            for timepoint_dir in timepoint_dirs:
                # Only look for .ome.zarr directories that match the stitcher output pattern
                zarr_files = []
                for item in timepoint_dir.iterdir():
                    if (item.is_dir() and 
                        item.name.endswith("_stitched.ome.zarr")):  # Strict stitcher output pattern
                        zarr_files.append(item)
                
                total_zarr_files += len(zarr_files)
                logging.info(f"Timepoint {timepoint_dir.name}: {len(zarr_files)} OME-ZARR files")
                
                # Extract region names and check for wellplate patterns (like A1, B2, etc.)
                for zarr_dir in zarr_files:
                    # Extract region name from directory name (same logic as HCS viewer)
                    region_name = zarr_dir.stem.replace("_stitched.ome", "")
                    
                    # Skip system files that start with ._
                    if region_name.startswith("._"):
                        continue
                    
                    # Check if region name matches wellplate pattern (letter followed by number)
                    # This is the key pattern the user mentioned: "one being a capitalized letter and the other a number"
                    wellplate_match = re.match(r'^([A-Z]+)(\d+)$', region_name)
                    if wellplate_match:
                        wellplate_regions.add(region_name)
                        logging.info(f"Found wellplate region: {region_name}")
            
            if total_zarr_files == 0:
                logging.info("No OME-ZARR files found in timepoint directories - not a wellplate dataset")
                return False
            
            # Determine if this is a wellplate dataset:
            # 1. Multiple timepoints with any zarr files, OR
            # 2. Single timepoint with multiple wellplate-pattern regions (A1, B2, etc.), OR  
            # 3. Single timepoint with multiple zarr files (fallback for non-standard naming)
            is_wellplate = (
                len(timepoint_dirs) > 1 or  # Multiple timepoints
                len(wellplate_regions) > 1 or  # Multiple wellplate regions
                (len(timepoint_dirs) == 1 and total_zarr_files > 1)  # Single timepoint with multiple files
            )
            
            if is_wellplate:
                wellplate_type = "multiple timepoints" if len(timepoint_dirs) > 1 else f"wellplate regions ({len(wellplate_regions)} wells)" if wellplate_regions else "multiple regions"
                logging.info(f"Wellplate dataset detected: {len(timepoint_dirs)} timepoints with {total_zarr_files} total files, type: {wellplate_type}")
                logging.info(f"Launching HCS Grid Viewer for wellplate visualization.")
            else:
                logging.info(f"Single output detected: {len(timepoint_dirs)} timepoints with {total_zarr_files} total files")
            
            return is_wellplate
            
        except Exception as e:
            logging.error(f"Error detecting wellplate dataset: {e}")
            return False

    def _launch_hcs_viewer(self, dataset_path: pathlib.Path) -> None:
        """Launch the HCS viewer for wellplate datasets"""
        try:
            import subprocess
            import sys
            
            # Find the HCS viewer script
            hcs_viewer_path = pathlib.Path(__file__).parent.parent / "hcs_viewer" / "run_grid_viewer.py"
            logging.info(f"HCS viewer script path: {hcs_viewer_path}")
            logging.info(f"HCS viewer script exists: {hcs_viewer_path.exists()}")
            logging.info(f"HCS viewer script is file: {hcs_viewer_path.is_file()}")
            
            if not hcs_viewer_path.exists():
                raise FileNotFoundError(f"HCS viewer script not found at: {hcs_viewer_path}")
            
            # Test if we can read the script
            try:
                with open(hcs_viewer_path, 'r') as f:
                    first_line = f.readline().strip()
                    logging.info(f"HCS viewer script first line: {first_line}")
            except Exception as e:
                logging.error(f"Cannot read HCS viewer script: {e}")
                raise
            
            # Verify dataset path exists
            logging.info(f"Dataset path: {dataset_path}")
            logging.info(f"Dataset path exists: {dataset_path.exists()}")
            logging.info(f"Dataset path is dir: {dataset_path.is_dir()}")
            
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
            
            # Launch the HCS viewer as a separate process
            cmd = [sys.executable, str(hcs_viewer_path), str(dataset_path)]
            logging.info(f"Executing command: {' '.join(cmd)}")
            
            # Set environment variables to help PyQt6 find the platform plugins
            env = os.environ.copy()
            
            # Try to find PyQt6 plugin path
            try:
                import PyQt6
                pyqt6_path = pathlib.Path(PyQt6.__file__).parent
                plugin_path = pyqt6_path / "Qt6" / "plugins"
                if plugin_path.exists():
                    env["QT_PLUGIN_PATH"] = str(plugin_path)
                    logging.info(f"Set QT_PLUGIN_PATH to: {plugin_path}")
                else:
                    logging.warning(f"PyQt6 plugin path not found at: {plugin_path}")
            except Exception as e:
                logging.warning(f"Could not set QT_PLUGIN_PATH: {e}")
            
            # Set additional Qt environment variables based on platform
            import platform
            system = platform.system().lower()
            if system == "darwin":  # macOS
                env["QT_QPA_PLATFORM"] = "cocoa"
                env["QT_MAC_WANTS_LAYER"] = "1"
            elif system == "linux":
                env["QT_QPA_PLATFORM"] = "xcb"
            elif system == "windows":
                env["QT_QPA_PLATFORM"] = "windows"
            # For other platforms, let Qt auto-detect
            
            # Use subprocess.run with capture_output to see any errors
            process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            import time
            time.sleep(2)

            # Check if process is still running
            if process.poll() is None:
                logging.info(f"HCS viewer launched successfully (PID: {process.pid})")
            else:
                # Process exited, check for errors
                stdout, stderr = process.communicate()
                logging.error(f"HCS viewer failed to start. Return code: {process.returncode}")
                logging.error(f"stdout: {stdout.decode()}")
                logging.error(f"stderr: {stderr.decode()}")
                raise Exception(f"HCS viewer failed to start: {stderr.decode()}")

        except Exception as e:
            logging.error(f"Failed to launch HCS viewer: {e}")
            logging.error("HCS viewer launch failed, raising exception to trigger fallback")
            # Re-raise the exception so the caller can handle the fallback
            raise
    
    def _open_in_napari(self, output_path: str) -> None:
        """Open output directly in Napari (original functionality)"""
        try:
            logging.info(f"Opening napari viewer for: {output_path}")
            
            # Create a new viewer instance each time
            viewer = napari.Viewer(show=True)
            
            if ".ome.zarr" in output_path:
                logging.info("Loading as OME-ZARR file")
                viewer.open(output_path, plugin="napari-ome-zarr")
            else:
                logging.info("Loading as regular file")
                viewer.open(output_path)

            # Apply color and contrast settings
            for layer in viewer.layers:
                wavelength = self.extractWavelength(layer.name)
                channel_info = CHANNEL_COLORS_MAP.get(
                    cast(Any, wavelength), {"hex": 0xFFFFFF, "name": "gray"}
                )

                # Set colormap
                if channel_info["name"] in AVAILABLE_COLORMAPS:
                    layer.colormap = AVAILABLE_COLORMAPS[channel_info["name"]]
                else:
                    layer.colormap = self.generateColormap(channel_info)

                # Set contrast limits based on dtype
                if np.issubdtype(layer.data.dtype, np.integer):
                    info = np.iinfo(layer.data.dtype)
                    layer.contrast_limits = (info.min, info.max)
                elif np.issubdtype(layer.data.dtype, np.floating):
                    layer.contrast_limits = (0.0, 1.0)

            # Don't call napari.run() - let the viewer run independently
            logging.info("Napari viewer opened successfully")
            
        except Exception as e:
            logging.error(f"An error occurred while opening output in Napari: {e}")
            import traceback
            traceback.print_exc()

    def extractWavelength(self, name: str) -> str | None:
        # Split the string and find the wavelength number immediately after "Fluorescence"
        parts = name.split()
        if "Fluorescence" in parts:
            index = parts.index("Fluorescence") + 1
            if index < len(parts):
                return parts[index].split()[0]  # Assuming '488 nm Ex' and taking '488'
        for color in ["R", "G", "B"]:
            if color in parts or "full_" + color in parts:
                return color
        return None

    def generateColormap(self, channel_info: dict[str, Any]) -> Colormap:
        """Convert a HEX value to a normalized RGB tuple."""
        c0 = (0, 0, 0)
        c1 = (
            ((channel_info["hex"] >> 16) & 0xFF) / 255,  # Normalize the Red component
            ((channel_info["hex"] >> 8) & 0xFF) / 255,  # Normalize the Green component
            (channel_info["hex"] & 0xFF) / 255,
        )  # Normalize the Blue component
        return Colormap(colors=[c0, c1], controls=[0, 1], name=channel_info["name"])


    def onSetFixedRegistration(self) -> None:
        """Handle Set Fixed Registration button click."""
        # TODO: TECHNICAL DEBT - Fixed registration is disabled pending characterization of 
        # X-Y stage motor drift patterns. The drift accumulation behavior (unidirectional, 
        # bidirectional, serpentine, etc.) must be determined for each specific microscope 
        # system before applying fixed shift corrections.
        
        QMessageBox.warning(
            self, 
            "Feature Disabled", 
            "Fixed registration is currently disabled.\n\n"
            "The drift pattern must be determined based on the specific X-Y stage motor "
            "characteristics of your microscope system.\n\n"
            "Please use standard registration instead, or contact the development team "
            "to characterize your system's drift behavior."
        )
        self.statusLabel.setText("Status: Fixed registration disabled (technical debt)")
        return
        
        # COMMENTED OUT - Re-enable after characterizing stage motor behavior
        # if not self.inputDirectory:
        #     QMessageBox.warning(self, "Input Error", "Please select an input directory.")
        #     return
        #     
        # try:
        #     from .registration.manual import launch_fixed_registration
        #     success = launch_fixed_registration(self.inputDirectory)
        #     if success:
        #         self.statusLabel.setText("Status: Fixed registration applied")
        #     else:
        #         self.statusLabel.setText("Status: Fixed registration cancelled")
        # except Exception as e:
        #     QMessageBox.critical(self, "Fixed Registration Error", str(e))
        #     self.statusLabel.setText("Status: Error in fixed registration")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    gui = StitchingGUI()
    gui.show()
    sys.exit(app.exec_())
