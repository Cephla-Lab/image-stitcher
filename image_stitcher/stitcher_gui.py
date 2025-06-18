import logging
import sys
import enum
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
    def __init__(self, params: StitchingParameters, perform_registration: bool, image_directory: str) -> None:
        super().__init__()
        self.params = params
        self.perform_registration = perform_registration
        self.image_directory = image_directory
        self.registration_complete = False
        self.registration_error = None
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
                    edge_width=256
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
    def __init__(self, image_directory: str, csv_path: str | None, output_csv_path: str | None) -> None:
        super().__init__()
        self.image_directory = image_directory
        self.csv_path = csv_path
        self.output_csv_path = output_csv_path

    def run(self) -> None:
        try:
            # Process all timepoints
            results = process_multiple_timepoints(
                base_directory=self.image_directory,
                overlap_diff_threshold=10,
                pou=3,
                ncc_threshold=0.5,
                edge_width=256
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
        self.registration_thread: RegistrationThread | None = None
        self.inputDirectory: str | None = (
            None  # This will be set by the directory selection
        )
        self.output_path = ""
        self.dtype: np.dtype | None = None
        self.flatfield_manifest: pathlib.Path | None = None
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

        # FOV Registration section (previously register button)
        self.registrationLabel = QLabel("FOV Registration:", self)
        self.mainLayout.addWidget(self.registrationLabel, 1, 0)
        self.registrationCombo = QComboBox(self)
        self.registrationCombo.addItems(["No", "Yes"])
        self.mainLayout.addWidget(self.registrationCombo, 1, 1)

        # Output format section
        self.outputFormatLabel = QLabel("Output Format:", self)
        self.mainLayout.addWidget(self.outputFormatLabel, 2, 0)
        self.outputFormatCombo = QComboBox(self)
        self.outputFormatCombo.addItems(["OME-ZARR"])  # Removed OME-TIFF as requested
        self.mainLayout.addWidget(self.outputFormatCombo, 2, 1)

        # Output compression section
        self.compressionLabel = QLabel("Output Compression:", self)
        self.mainLayout.addWidget(self.compressionLabel, 3, 0)
        self.outputCompression = QComboBox(self)
        self.outputCompression.addItems(["default", "none"])
        self.mainLayout.addWidget(self.outputCompression, 3, 1)

        # Pyramid checkbox section - now with checkbox on the right
        self.pyramidLabel = QLabel("Infer levels for output image pyramid:", self)
        self.mainLayout.addWidget(self.pyramidLabel, 4, 0)
        self.pyramidCheckbox = QCheckBox(self)
        self.pyramidCheckbox.setChecked(True)
        self.mainLayout.addWidget(self.pyramidCheckbox, 4, 1)

        # Flatfield Correction section
        self.flatfieldModeLabel = QLabel("Correction Mode:", self)
        self.mainLayout.addWidget(self.flatfieldModeLabel, 5, 0)
        self.flatfieldModeCombo = QComboBox(self)
        self.flatfieldModeCombo.addItems(
            [
                "No Flatfield Correction",
                "Compute Flatfield Correction",
                "Load Precomputed Flatfield",
            ]
        )
        self.flatfieldModeCombo.currentIndexChanged.connect(self.onFlatfieldModeChanged)
        self.mainLayout.addWidget(self.flatfieldModeCombo, 5, 1)

        self.flatfieldLoadLabel = QLabel("Load Flatfield:", self)
        self.mainLayout.addWidget(self.flatfieldLoadLabel, 6, 0)
        self.flatfieldLoadLabel.setVisible(False) # Initially hidden
        self.loadFlatfieldDropArea = DragDropArea("Drag & Drop Flatfield Directory Here", self)
        self.loadFlatfieldDropArea.path_dropped.connect(self.onLoadFlatfieldDropped)
        self.mainLayout.addWidget(self.loadFlatfieldDropArea, 6, 1)
        self.loadFlatfieldDropArea.setVisible(False) # Initially hidden

        # Z-Stack Options section
        self.zLayerLabel = QLabel("Processing Mode:", self)
        self.mainLayout.addWidget(self.zLayerLabel, 7, 0)
        self.zLayerModeCombo = QComboBox(self)
        self.zLayerModeCombo.addItems(["Middle Layer", "All Layers", "Specific Layer", "Maximum Intensity Projection (MIP)"])
        self.zLayerModeCombo.currentIndexChanged.connect(self.onZLayerModeChanged)
        self.mainLayout.addWidget(self.zLayerModeCombo, 7, 1)

        self.zLayerSpinLabel = QLabel("Select Z-Layer Index:", self)
        self.mainLayout.addWidget(self.zLayerSpinLabel, 8, 0)
        self.zLayerSpinLabel.setVisible(False) 

        self.zLayerSpinBox = QSpinBox(self)
        self.zLayerSpinBox.setMinimum(0)
        self.zLayerSpinBox.setMaximum(999)  # Will be updated based on actual data
        self.mainLayout.addWidget(self.zLayerSpinBox, 8, 1)
        self.zLayerSpinBox.setVisible(False) 

        # Status and Progress section
        self.statusLabel = QLabel("Status: Ready", self)
        self.mainLayout.addWidget(self.statusLabel, 9, 0, 1, 2)  

        self.progressBar = QProgressBar(self)
        self.progressBar.hide()
        self.mainLayout.addWidget(self.progressBar, 10, 0, 1, 2)  

        # Action Buttons
        self.startBtn = QPushButton("Start Stitching", self)
        self.startBtn.clicked.connect(self.onStitchingStart)
        self.mainLayout.addWidget(self.startBtn, 11, 0)

        self.viewBtn = QPushButton("View Output in Napari", self)
        self.viewBtn.clicked.connect(self.onViewOutput)
        self.viewBtn.setEnabled(False)
        self.mainLayout.addWidget(self.viewBtn, 11, 1)
        
        # Add stretch to push everything to the top
        self.mainLayout.setRowStretch(12, 1) 

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
        # Visually mark the drop area
        self.inputDirDropArea.setText(f"Loaded: {acquisition_dir.name}")
        self.inputDirDropArea.setStyleSheet("""
            QLabel {border: 2px solid green; border-radius: 5px; background-color: #e0ffe0;}
        """)
        self.probeDatasetForZLayers()

    def selectInputDirectory(self) -> None: # Kept for now, can be removed if button is fully replaced
        dir = QFileDialog.getExistingDirectory(self, "Select Input Image Folder")
        if dir:
            self.inputDirectory = dir
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
        """Start stitching from GUI."""
        if not self.inputDirectory:
            QMessageBox.warning(
                self, "Input Error", "Please select an input directory."
            )
            return

        try:
            # Create parameters from UI state
            format_text = self.outputFormatCombo.currentText()
            if format_text == "OME-ZARR":
                output_format = OutputFormat.ome_zarr
            else:
                QMessageBox.critical(self, "Internal Error", f"Invalid output format selected: {format_text}")
                return

            flatfield_mode = get_flatfield_mode_from_string(self.flatfieldModeCombo.currentText())
            apply_flatfield = flatfield_mode != FlatfieldModeOption.NONE
            flatfield_manifest = self.flatfield_manifest if flatfield_mode == FlatfieldModeOption.LOAD else None

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

            # Check if registration is requested
            perform_registration = self.registrationCombo.currentText() == "Yes"

            # Create and configure the stitcher thread
            self.stitcher = StitcherThread(
                params=params,
                perform_registration=perform_registration,
                image_directory=self.inputDirectory
            )

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
            if perform_registration:
                self.statusLabel.setText("Status: Registering images...")
            else:
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
        QMessageBox.information(
            self,
            "Stitching Complete",
            "Image stitching has been completed successfully."
        )

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

    def onViewOutput(self) -> None:
        output_path = self.output_path
        if not output_path:
            QMessageBox.warning(self, "View Error", "No output path set. Has stitching completed?")
            return
        try:
            viewer = napari.Viewer()
            if ".ome.zarr" in output_path:
                viewer.open(output_path, plugin="napari-ome-zarr")
            else:
                viewer.open(output_path)

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

            napari.run()
        except Exception as e:
            QMessageBox.critical(self, "Error Opening in Napari", str(e))
            logging.error(f"An error occurred while opening output in Napari: {e}")

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

    def onRegistrationStart(self) -> None:
        """Start registration from GUI."""
        if not self.inputDirectory:
            QMessageBox.warning(
                self, "Input Error", "Please select an input directory."
            )
            return

        try:
            # Start registration in a separate thread
            self.registration_thread = RegistrationThread(
                image_directory=self.inputDirectory,
                csv_path=None,  # Not needed as process_multiple_timepoints handles this
                output_csv_path=None  # Not needed as process_multiple_timepoints handles this
            )
            self.registration_thread.error.connect(self.onRegistrationError)
            self.registration_thread.finished.connect(self.onRegistrationFinished)

            # Update UI
            self.statusLabel.setText("Status: Registering images...")
            self.progressBar.setRange(0, 0)  # Indeterminate progress
            self.progressBar.show()

            # Start registration
            self.registration_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Registration Error", str(e))
            self.statusLabel.setText("Status: Error Encountered")

    def onRegistrationError(self, error_msg: str) -> None:
        """Handle registration errors."""
        QMessageBox.critical(self, "Registration Error", error_msg)
        self.statusLabel.setText("Status: Error Encountered")
        self.progressBar.hide()

    def onRegistrationFinished(self) -> None:
        """Handle completion of registration."""
        self.statusLabel.setText("Status: Registration completed")
        self.progressBar.hide()
        QMessageBox.information(
            self,
            "Registration Complete",
            "Image registration has been completed successfully."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    gui = StitchingGUI()
    gui.show()
    sys.exit(app.exec_())
