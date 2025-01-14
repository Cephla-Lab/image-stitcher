import sys
from typing import Any, cast

import napari
import numpy as np
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
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
        self.initUI()

    def initUI(self) -> None:
        self.layout = QVBoxLayout(self)  # type: ignore

        # Input Directory Selection (full width at the top)
        self.inputDirectoryBtn = QPushButton("Select Input Directory", self)
        self.inputDirectoryBtn.clicked.connect(self.selectInputDirectory)
        self.layout.addWidget(self.inputDirectoryBtn)  # type: ignore

        # Create a horizontal layout for checkboxes and registration inputs
        _ = QHBoxLayout()

        # Left side: Checkboxes
        grid = QGridLayout()

        self.applyFlatfieldCheck = QCheckBox("Flatfield Correction", self)
        grid.addWidget(self.applyFlatfieldCheck, 0, 0)

        self.useRegistrationCheck = QCheckBox("Cross-Correlation Registration", self)
        self.useRegistrationCheck.toggled.connect(self.onRegistrationCheck)
        grid.addWidget(self.useRegistrationCheck, 1, 0)

        self.mergeTimepointsCheck = QCheckBox(
            "Merge Timepoints", self
        )  # MERGE_TIMEPOINTS
        self.mergeTimepointsCheck.setChecked(False)  # Default to False
        grid.addWidget(self.mergeTimepointsCheck, 2, 0)

        self.mergeRegionsCheck = QCheckBox(
            "Merge HCS Regions", self
        )  # STITCH_COMPLETE_ACQUISITION
        self.mergeRegionsCheck.setChecked(False)  # Default to False
        grid.addWidget(self.mergeRegionsCheck, 3, 0)

        # Right side: Registration inputs
        self.channelLabel = QLabel("Registration Channel", self)
        self.channelLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore
        self.channelCombo = QComboBox(self)
        grid.addWidget(self.channelLabel, 0, 2)
        grid.addWidget(self.channelCombo, 0, 3)

        self.zLevelLabel = QLabel("Registration Z-Level", self)
        self.zLevelLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore
        self.zLevelInput = QSpinBox(self)
        self.zLevelInput.setMinimum(0)
        self.zLevelInput.setMaximum(100)
        grid.addWidget(self.zLevelLabel, 1, 2)
        grid.addWidget(self.zLevelInput, 1, 3)

        self.layout.addLayout(grid)  # type: ignore

        # Output format combo box (full width)
        self.outputFormatCombo = QComboBox()
        self.outputFormatCombo.addItems(["OME-ZARR", "OME-TIFF"])
        self.layout.addWidget(self.outputFormatCombo)  # type: ignore
        self.outputFormatCombo.currentTextChanged.connect(self.onOutputFormatChanged)

        # Status label
        self.statusLabel = QLabel("Status: Ready", self)
        self.layout.addWidget(self.statusLabel)  # type: ignore

        # Start stitching button
        self.startBtn = QPushButton("Start Stitching", self)
        self.startBtn.clicked.connect(self.onStitchingStart)
        self.layout.addWidget(self.startBtn)  # type: ignore

        # Progress bar setup
        self.progressBar = QProgressBar(self)
        self.progressBar.hide()
        self.layout.addWidget(self.progressBar)  # type: ignore

        # Output path QLineEdit
        self.outputPathEdit = QLineEdit(self)
        self.outputPathEdit.setPlaceholderText(
            "Enter Filepath To Visualize (No Stitching Required)"
        )
        self.layout.addWidget(self.outputPathEdit)  # type: ignore

        # View in Napari button
        self.viewBtn = QPushButton("View Output in Napari", self)
        self.viewBtn.clicked.connect(self.onViewOutput)
        self.viewBtn.setEnabled(False)
        self.layout.addWidget(self.viewBtn)  # type: ignore

        self.setWindowTitle("Cephla Image Stitcher")
        self.setGeometry(300, 300, 500, 300)
        self.show()

        # Initially hide registration inputs
        self.zLevelLabel.hide()
        self.zLevelInput.hide()
        self.channelLabel.hide()
        self.channelCombo.hide()

    def selectInputDirectory(self) -> None:
        self.inputDirectory = QFileDialog.getExistingDirectory(
            self, "Select Input Image Folder"
        )
        if self.inputDirectory:
            self.inputDirectoryBtn.setText(f"Selected: {self.inputDirectory}")
            self.onRegistrationCheck(self.useRegistrationCheck.isChecked())

    def onRegistrationCheck(self, checked: Any) -> None:
        """Handle registration checkbox state change."""
        self.zLevelLabel.setVisible(checked)
        self.zLevelInput.setVisible(checked)
        self.channelLabel.setVisible(checked)
        self.channelCombo.setVisible(checked)

        if checked:
            if not self.inputDirectory:
                QMessageBox.warning(
                    self, "Input Error", "Please Select an Input Image Folder First"
                )
                self.useRegistrationCheck.setChecked(False)
                return

            try:
                # Create temporary params with minimal required parameters
                temp_params = StitchingParameters(input_folder=self.inputDirectory)

                # Create temporary Stitcher to parse filenames
                stitcher = Stitcher(temp_params)
                timepoints = stitcher.computed_parameters.timepoints

                if not timepoints:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "No time points found in the selected directory.",
                    )
                    self.useRegistrationCheck.setChecked(False)
                    return

                # Setup Z-Level
                self.zLevelInput.setMinimum(0)
                self.zLevelInput.setMaximum(stitcher.computed_parameters.num_z - 1)

                # Setup channel dropdown
                self.channelCombo.clear()
                self.channelCombo.addItems(stitcher.computed_parameters.channel_names)

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Parsing Error",
                    f"An error occurred during data processing: {e}",
                )
                self.useRegistrationCheck.setChecked(False)
                self.zLevelLabel.hide()
                self.zLevelInput.hide()
                self.channelLabel.hide()
                self.channelCombo.hide()

    def onStitchingStart(self) -> None:
        """Start stitching from GUI."""
        if not self.inputDirectory:
            QMessageBox.warning(
                self, "Input Error", "Please select an input directory."
            )
            return

        # # In StitchingGUI.onStitchingStart():
        # if self.outputFormatCombo.currentText() == 'OME-TIFF' and (self.mergeTimepointsCheck.isChecked() or self.mergeRegionsCheck.isChecked()):
        #     QMessageBox.warning(self, "Format Warning",
        #                        "Merging operations are only supported for OME-ZARR format. "
        #                        "These operations will be skipped.")

        try:
            # Create parameters from UI state
            params = StitchingParameters(
                input_folder=self.inputDirectory,
                output_format=OutputFormat(
                    "." + self.outputFormatCombo.currentText().lower().replace("-", ".")
                ),
                apply_flatfield=self.applyFlatfieldCheck.isChecked(),
                use_registration=self.useRegistrationCheck.isChecked(),
                registration_channel=self.channelCombo.currentText()
                if self.useRegistrationCheck.isChecked()
                else None,
                registration_z_level=self.zLevelInput.value()
                if self.useRegistrationCheck.isChecked()
                else 0,
                scan_pattern=ScanPattern.unidirectional,
                merge_timepoints=self.mergeTimepointsCheck.isChecked(),
                merge_hcs_regions=self.mergeRegionsCheck.isChecked(),
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
        self.outputPathEdit.setText(path)
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

    def onOutputFormatChanged(self) -> None:
        is_zarr = self.outputFormatCombo.currentText() == "OME-ZARR"
        self.mergeTimepointsCheck.setEnabled(is_zarr)
        self.mergeRegionsCheck.setEnabled(is_zarr)
        if not is_zarr:
            self.mergeTimepointsCheck.setChecked(False)
            self.mergeRegionsCheck.setChecked(False)

    def onViewOutput(self) -> None:
        output_path = self.outputPathEdit.text()
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
            print(f"An error occurred while opening output in Napari: {e}")

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
    app = QApplication(sys.argv)
    gui = StitchingGUI()
    gui.show()
    sys.exit(app.exec_())
