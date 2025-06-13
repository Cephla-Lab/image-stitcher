import logging
import sys
from typing import Any, cast
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
)

from .parameters import OutputFormat, ScanPattern, StitchingParameters
from .stitcher import ProgressCallbacks, Stitcher

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
    def __init__(self, inner: Stitcher) -> None:
        super().__init__()
        self.inner = inner

    def run(self) -> None:
        self.inner.run()


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
        self.initUI()

    def initUI(self) -> None:
        self.mainLayout = QGridLayout(self) # Main layout for the window
        self.setLayout(self.mainLayout)
        self.mainLayout.setSpacing(10)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)

        # --- Acquisition Settings --- #
        acquisition_group = QGroupBox("Acquisition Settings")
        acquisition_layout = QGridLayout()
        acquisition_group.setLayout(acquisition_layout)
        self.mainLayout.addWidget(acquisition_group, 0, 0, 1, 2)

        self.inputDirLabel = QLabel("Acquisition Directory:", self)
        acquisition_layout.addWidget(self.inputDirLabel, 0, 0)
        self.inputDirDropArea = DragDropArea("Drag & Drop Input Directory Here", self)
        self.inputDirDropArea.path_dropped.connect(self.onInputDirectoryDropped)
        acquisition_layout.addWidget(self.inputDirDropArea, 0, 1)

        # --- Flatfield Correction Options --- #
        flatfield_group = QGroupBox("Flatfield Correction")
        flatfield_layout = QGridLayout()
        flatfield_group.setLayout(flatfield_layout)
        self.mainLayout.addWidget(flatfield_group, 1, 0, 1, 2) 

        self.flatfieldModeLabel = QLabel("Correction Mode:", self)
        flatfield_layout.addWidget(self.flatfieldModeLabel, 0, 0)
        self.flatfieldModeCombo = QComboBox(self)
        self.flatfieldModeCombo.addItems(
            [
                "No Flatfield Correction",
                "Compute Flatfield Correction",
                "Load Precomputed Flatfield",
            ]
        )
        self.flatfieldModeCombo.currentIndexChanged.connect(self.onFlatfieldModeChanged)
        flatfield_layout.addWidget(self.flatfieldModeCombo, 0, 1)

        self.flatfieldLoadLabel = QLabel("Load Flatfield:", self)
        flatfield_layout.addWidget(self.flatfieldLoadLabel, 1, 0)
        self.flatfieldLoadLabel.setVisible(False) # Initially hidden
        self.loadFlatfieldDropArea = DragDropArea("Drag & Drop Flatfield Directory Here", self)
        self.loadFlatfieldDropArea.path_dropped.connect(self.onLoadFlatfieldDropped)
        flatfield_layout.addWidget(self.loadFlatfieldDropArea, 1, 1)
        self.loadFlatfieldDropArea.setVisible(False) # Initially hidden

        # --- Z-Stack Options --- #        
        z_stack_group = QGroupBox("Z-Stack Options")
        z_stack_layout = QGridLayout()
        z_stack_group.setLayout(z_stack_layout)
        self.mainLayout.addWidget(z_stack_group, 2, 0, 1, 2)

        self.zLayerLabel = QLabel("Processing Mode:", self)
        z_stack_layout.addWidget(self.zLayerLabel, 0, 0)
        self.zLayerModeCombo = QComboBox(self)
        self.zLayerModeCombo.addItems(["Middle Layer", "All Layers", "Specific Layer", "Maximum Intensity Projection (MIP)"])
        self.zLayerModeCombo.currentIndexChanged.connect(self.onZLayerModeChanged)
        z_stack_layout.addWidget(self.zLayerModeCombo, 0, 1)

        self.zLayerSpinLabel = QLabel("Select Z-Layer Index:", self)
        z_stack_layout.addWidget(self.zLayerSpinLabel, 1, 0)
        self.zLayerSpinLabel.setVisible(False) 

        self.zLayerSpinBox = QSpinBox(self)
        self.zLayerSpinBox.setMinimum(0)
        self.zLayerSpinBox.setMaximum(999)  # Will be updated based on actual data
        z_stack_layout.addWidget(self.zLayerSpinBox, 1, 1)
        self.zLayerSpinBox.setVisible(False) 

        # --- Status and Progress --- # 
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout() # Using QVBoxLayout for simple vertical stacking
        status_group.setLayout(status_layout)
        self.mainLayout.addWidget(status_group, 3, 0, 1, 2)

        self.statusLabel = QLabel("Status: Ready", self)
        status_layout.addWidget(self.statusLabel)  

        self.progressBar = QProgressBar(self)
        self.progressBar.hide()
        status_layout.addWidget(self.progressBar)  

        # --- Action Buttons --- #
        # Start stitching button
        self.startBtn = QPushButton("Start Stitching", self)
        self.startBtn.clicked.connect(self.onStitchingStart)
        self.mainLayout.addWidget(self.startBtn, 4, 0, 1, 1)

        # View in Napari button
        self.viewBtn = QPushButton("View Output in Napari", self)
        self.viewBtn.clicked.connect(self.onViewOutput)
        self.viewBtn.setEnabled(False)
        self.mainLayout.addWidget(self.viewBtn, 4, 1, 1, 1)
        
        # Add stretch to push everything to the top
        self.mainLayout.setRowStretch(5, 1) 

        self.setWindowTitle("Cephla Image Stitcher")
        self.setGeometry(300, 300, 600, 400) # Adjusted size
        self.show()

    def onInputDirectoryDropped(self, path: str) -> None:
        if pathlib.Path(path).is_dir():
            self.inputDirectory = path
            self.probeDatasetForZLayers()
        else:
            QMessageBox.warning(self, "Input Error", "Please drop a directory, not a file.")
            self.inputDirDropArea.setText("Drag & Drop Input Directory Here") # Reset text
            self.inputDirDropArea.setStyleSheet("""
                QLabel {
                    border: 2px dashed #aaa;
                    border-radius: 5px;
                    background-color: #f0f0f0;
                }
            """)

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
            mode = self.flatfieldModeCombo.currentIndex()
            apply_flatfield = mode in (1, 2)
            flatfield_manifest = self.flatfield_manifest if mode == 2 else None

            # Determine z-layer selection strategy
            z_layer_mode = self.zLayerModeCombo.currentIndex()
            if z_layer_mode == 0:  # Middle Layer
                z_layer_selection = "middle"
            elif z_layer_mode == 1:  # All Layers
                z_layer_selection = "all"
            elif z_layer_mode == 2:  # Specific Layer
                z_layer_selection = str(self.zLayerSpinBox.value())
            else:  # MIP
                z_layer_selection = "mip"

            params = StitchingParameters(
                input_folder=self.inputDirectory,
                output_format=OutputFormat.ome_zarr,
                scan_pattern=ScanPattern.unidirectional,
                apply_flatfield=apply_flatfield,
                flatfield_manifest=flatfield_manifest,
                z_layer_selection=z_layer_selection,
            )

            self.stitcher = StitcherThread(
                Stitcher(
                    params,
                    ProgressCallbacks(
                        update_progress=self.update_progress.emit,
                        getting_flatfields=self.getting_flatfields.emit,
                        starting_stitching=self.starting_stitching.emit,
                        starting_saving=self.starting_saving.emit,
                        finished_saving=self.finished_saving.emit,
                    ),
                )
            )
            self.setupConnections()

            # Start processing
            self.statusLabel.setText("Status: Stitching...")
            self.stitcher.start()
            self.progressBar.show()

        except Exception as e:
            QMessageBox.critical(self, "Stitching Error", str(e))
            self.statusLabel.setText("Status: Error Encountered")

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    gui = StitchingGUI()
    gui.show()
    sys.exit(app.exec_())
