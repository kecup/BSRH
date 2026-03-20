import os
import sys
import threading
import time
import math
import zstandard as zstd
import dill as pickle
import numpy as np
import datetime as dt
import traceback
from scipy.optimize import curve_fit
from scipy.signal import convolve, get_window

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QProgressBar, QMessageBox, QDialog,
    QListWidget, QListWidgetItem, QGroupBox, QFormLayout, QComboBox, QCheckBox,
    QTabWidget, QSplitter, QDialogButtonBox, QScrollArea, QSlider
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

class DirectoryEntryWidget(QWidget):
    def __init__(self, directory, total_files, parent=None):
        super().__init__(parent)
        self.directory = directory
        self.total_files = total_files
        self.start_idx = 0
        self.end_idx = max(0, total_files - 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with path and remove button
        header_layout = QHBoxLayout()
        path_label = QLabel(directory)
        path_label.setWordWrap(True)
        self.remove_btn = QPushButton("Remove")
        header_layout.addWidget(path_label, stretch=1)
        header_layout.addWidget(self.remove_btn)
        layout.addLayout(header_layout)

        # Sliders layout
        slider_layout = QFormLayout()

        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.setRange(0, self.end_idx)
        self.start_slider.setValue(self.start_idx)
        self.start_label = QLabel(f"Start: 0")

        self.end_slider = QSlider(Qt.Horizontal)
        self.end_slider.setRange(0, self.end_idx)
        self.end_slider.setValue(self.end_idx)
        self.end_label = QLabel(f"End: {self.end_idx}")

        slider_layout.addRow(self.start_label, self.start_slider)
        slider_layout.addRow(self.end_label, self.end_slider)
        layout.addLayout(slider_layout)

        # Connections
        self.start_slider.valueChanged.connect(self.on_start_changed)
        self.end_slider.valueChanged.connect(self.on_end_changed)

    def on_start_changed(self, val):
        if val > self.end_slider.value():
            self.start_slider.setValue(self.end_slider.value())
            val = self.end_slider.value()
        self.start_idx = val
        self.start_label.setText(f"Start: {val}")

    def on_end_changed(self, val):
        if val < self.start_slider.value():
            self.end_slider.setValue(self.start_slider.value())
            val = self.start_slider.value()
        self.end_idx = val
        self.end_label.setText(f"End: {val}")

    def get_range(self):
        return self.directory, self.start_idx, self.end_idx

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:
    import pytimber
    HAS_PYTIMBER = True
except ImportError:
    HAS_PYTIMBER = False

import matplotlib.patches as patches

class PointDetailsDialog(QDialog):
    def __init__(self, filepath, event_idx, plane, maskX, maskY, maskR, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Point Details: {os.path.basename(filepath)} [Event {event_idx}]")
        self.resize(800, 600)

        layout = QVBoxLayout(self)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        # Load the data
        try:
            dctx = zstd.ZstdDecompressor()
            with open(filepath, 'rb') as f:
                with dctx.stream_reader(f) as reader:
                    file_data = pickle.load(reader)

            event = file_data[event_idx]
            is_1d_profiles = False
            image_2d = None
            profile_h = None
            profile_v = None

            if isinstance(event.cam_data, list) and len(event.cam_data) >= 2:
                is_1d_profiles = True
                profile_h, profile_v = event.cam_data[0], event.cam_data[1]
            elif isinstance(event.cam_data, dict) and 'image2D' in event.cam_data:
                img_data = event.cam_data['image2D']
                if isinstance(img_data, list) and len(img_data) >= 2:
                    is_1d_profiles = True
                    profile_h, profile_v = img_data[0], img_data[1]
                elif hasattr(img_data, 'shape') and img_data.shape[0] <= 2:
                    is_1d_profiles = True
                    profile_h = img_data[0]
                    if img_data.shape[0] > 1:
                        profile_v = img_data[1]
                    else:
                        profile_v = np.zeros_like(profile_h)
                else:
                    image_2d = img_data

            if not is_1d_profiles and image_2d is not None:
                # Show 2D image and circle
                self.ax.imshow(image_2d, cmap='jet')
                circle = patches.Circle((maskX, maskY), maskR, color='r', fill=False)
                self.ax.add_patch(circle)
                self.ax.set_title("2D Raw Image with Mask")
            elif is_1d_profiles:
                # Show 1D profile
                profile = profile_h if plane == 'H' else profile_v
                self.ax.plot(profile, color='blue')
                mask_center = maskX if plane == 'H' else maskY
                self.ax.axvline(x=mask_center - maskR, color='r', linestyle='--')
                self.ax.axvline(x=mask_center + maskR, color='r', linestyle='--')
                self.ax.set_title(f"1D Profile ({plane}) with Mask Boundaries")
            else:
                self.ax.text(0.5, 0.5, "Unknown data format", ha='center', va='center')

        except Exception as e:
            self.ax.text(0.5, 0.5, f"Error loading event:\n{e}", ha='center', va='center')
            traceback.print_exc()

        self.canvas.draw()

class MaskValidationDialog(QDialog):
    def __init__(self, image, initial_x, initial_y, initial_r, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Validate Mask Position")
        self.resize(800, 600)

        self.image = image
        self.maskX = initial_x
        self.maskY = initial_y
        self.maskR = initial_r

        layout = QVBoxLayout(self)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        controls = QFormLayout()

        h, w = self.image.shape[:2]

        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setRange(0, w)
        self.x_slider.setValue(self.maskX)
        self.x_label = QLabel(f"Mask X: {self.maskX}")

        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setRange(0, h)
        self.y_slider.setValue(self.maskY)
        self.y_label = QLabel(f"Mask Y: {self.maskY}")

        self.r_slider = QSlider(Qt.Horizontal)
        self.r_slider.setRange(1, 100)
        self.r_slider.setValue(self.maskR)
        self.r_label = QLabel(f"Mask R: {self.maskR}")

        controls.addRow(self.x_label, self.x_slider)
        controls.addRow(self.y_label, self.y_slider)
        controls.addRow(self.r_label, self.r_slider)

        self.x_slider.valueChanged.connect(self.update_plot)
        self.y_slider.valueChanged.connect(self.update_plot)
        self.r_slider.valueChanged.connect(self.update_plot)

        layout.addLayout(controls)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.circle = None
        self.initial_plot()

    def initial_plot(self):
        self.ax.imshow(self.image, cmap='jet')
        self.circle = patches.Circle((self.maskX, self.maskY), self.maskR, color='r', fill=False)
        self.ax.add_patch(self.circle)
        self.canvas.draw()

    def update_plot(self):
        self.maskX = self.x_slider.value()
        self.maskY = self.y_slider.value()
        self.maskR = self.r_slider.value()

        self.x_label.setText(f"Mask X: {self.maskX}")
        self.y_label.setText(f"Mask Y: {self.maskY}")
        self.r_label.setText(f"Mask R: {self.maskR}")

        if self.circle:
            self.circle.center = (self.maskX, self.maskY)
            self.circle.set_radius(self.maskR)
            self.canvas.draw()

    def get_mask_params(self):
        return self.maskX, self.maskY, self.maskR

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

class AnalysisWorker(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, task_type, directories=None, params=None, bws_args=None, parent=None):
        super().__init__(parent)
        self.task_type = task_type
        self.directories = directories
        self.params = params
        self.bws_args = bws_args

        self.results = {
            'H': {'bsrh': {}, 'bws': {}},
            'V': {'bsrh': {}, 'bws': {}}
        }
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            if self.task_type == 'BSRH':
                self.run_bsrh_analysis()
            elif self.task_type == 'BWS':
                self.fetch_bws_data()
            self.finished_signal.emit(self.results)
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))

    def run_bsrh_analysis(self):
        sigma = self.params['sigma']
        pixelSizeH = self.params['pixelSizeH']
        pixelSizeV = self.params['pixelSizeV']
        psfWidthH = self.params['psfWidthH']
        psfWidthV = self.params['psfWidthV']
        rl_iterations = int(self.params['rl_iterations'])

        # Hardcoded Constants
        INTENSIFIER_GAIN = 5651.0

        SIGMA_TO_MM_X = 263.0
        SIGMA_TO_MM_Y = 361.0

        y1, y2 = 154, 454
        x1, x2 = 284, 684

        dctx = zstd.ZstdDecompressor()

        total_files = 0

        # Track cPc per bunch dynamically from unmasked background images
        cpc_per_bunch = {}
        # Fallback if an unmasked image hasn't been seen for a bunch yet
        global_cpc_fallback = 44.0
        all_files = []
        for d, start_idx, end_idx in self.directories:
            if os.path.isdir(d):
                files = [f for f in os.listdir(d) if f.endswith('.pkl.zst')]
                files = sorted(files)
                # Apply range
                end_idx = min(end_idx, len(files) - 1)
                if start_idx <= end_idx and start_idx < len(files):
                    files_in_range = files[start_idx:end_idx+1]
                    all_files.extend([os.path.join(d, f) for f in files_in_range])

        total_files = len(all_files)

        if total_files == 0:
            self.error_signal.emit("No .pkl.zst files found in selected directories/ranges.")
            return

        # 1. Find the first masked image to auto-detect and validate mask
        maskX, maskY, maskR = self.params.get('maskX', 455), self.params.get('maskY', 300), self.params.get('maskR', 40)

        for i, filepath in enumerate(all_files):
            if not self.is_running:
                return
            self.progress_signal.emit(int((i / total_files) * 100))
            self.status_signal.emit(f"Processing file {i+1}/{total_files}: {os.path.basename(filepath)}")

            try:
                with open(filepath, 'rb') as f:
                    with dctx.stream_reader(f) as reader:
                        file_data = pickle.load(reader)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            for event_idx, event in enumerate(file_data):
                if not self.is_running:
                    return
                try:
                    timestamp = event.timestamp
                    if timestamp.tzinfo is not None:
                        timestamp = timestamp.replace(tzinfo=None)

                    delaySlots_raw = event.device_data.get("GATE-INT", {}).get("delaySlots", -1)
                    delaySlots = int(np.asarray(delaySlots_raw).item()) if isinstance(delaySlots_raw, np.ndarray) else int(delaySlots_raw)
                    if delaySlots == -1:
                        continue

                    tmpFBCT = event.device_data.get('FBCT1', {}).get('averageBunchIntensities', [])
                    if isinstance(tmpFBCT, np.ndarray) and tmpFBCT.size == 0:
                        tmpFBCT = []
                    bunch_charge = 0
                    if len(tmpFBCT) > delaySlots:
                        # Attempt to handle pulseLength if present (like in cpc_bunches.py)
                        pulse_length_raw = event.device_data.get("GATE-INT", {}).get("pulseLength", 4)
                        pulse_length = int(np.asarray(pulse_length_raw).item()) if isinstance(pulse_length_raw, np.ndarray) else int(pulse_length_raw)

                        end_slot = min(len(tmpFBCT), delaySlots + max(1, pulse_length // 4))
                        bunch_charge = np.sum(tmpFBCT[delaySlots:end_slot])

                    # Fix truth value of an array
                    if isinstance(bunch_charge, np.ndarray):
                        if bunch_charge.size == 1:
                            bunch_charge = bunch_charge.item()
                        else:
                            bunch_charge = np.sum(bunch_charge)

                    if bunch_charge <= 0:
                        continue # Can't calculate relative projection without a reference FBCT charge

                    is_1d_profiles = False
                    profile_h, profile_v = None, None
                    image_2d = None

                    if isinstance(event.cam_data, list) and len(event.cam_data) >= 2:
                        is_1d_profiles = True
                        profile_h, profile_v = event.cam_data[0], event.cam_data[1]
                    elif isinstance(event.cam_data, dict) and 'image2D' in event.cam_data:
                        img_data = event.cam_data['image2D']
                        if isinstance(img_data, list) and len(img_data) >= 2:
                            is_1d_profiles = True
                            profile_h, profile_v = img_data[0], img_data[1]
                        elif hasattr(img_data, 'shape') and img_data.shape[0] <= 2:
                            is_1d_profiles = True
                            profile_h = img_data[0]
                            if img_data.shape[0] > 1:
                                profile_v = img_data[1]
                            else:
                                profile_v = np.zeros_like(profile_h)
                        else:
                            image_2d = img_data

                    if not is_1d_profiles and image_2d is None:
                        continue # Unrecognized data format

                    exposure_raw = event.device_data.get('GATE-INT', {}).get('numberOfPulses', 1)
                    exposure = int(np.asarray(exposure_raw).item()) if isinstance(exposure_raw, np.ndarray) else int(exposure_raw)
                    if exposure == 0: exposure = 1

                    fArr = event.device_data.get('FILTERS.B2', {}).get('channelState', [0,0,0])
                    # Ensure scalars for the math
                    fArr0 = np.asarray(fArr[0]).item() if isinstance(fArr[0], np.ndarray) else fArr[0]
                    fArr1 = np.asarray(fArr[1]).item() if isinstance(fArr[1], np.ndarray) else fArr[1]
                    fArr2 = np.asarray(fArr[2]).item() if isinstance(fArr[2], np.ndarray) else fArr[2]
                    in_filter = pow(12.6, 3 * fArr0 + 2 * fArr1 + fArr2)

                    cmask_x = event.device_data.get('CMASK-X', None)
                    cmask_y = event.device_data.get('CMASK-Y', None)

                    # Assuming mask is present if CMASK is close to mask position
                    # "different CMASK-X or Y value" for background
                    # Let's say if abs(cmask_x - maskX) < some_threshold, it's masked
                    is_bg_H = False
                    if cmask_x is not None:
                        cmask_x_val = np.asarray(cmask_x).item() if isinstance(cmask_x, np.ndarray) else cmask_x
                        if abs(cmask_x_val - maskX) > 100:
                            is_bg_H = True
                    # If there's no CMASK info, assume masked for safety

                    is_bg_V = False
                    if cmask_y is not None:
                        cmask_y_val = np.asarray(cmask_y).item() if isinstance(cmask_y, np.ndarray) else cmask_y
                        if abs(cmask_y_val - maskY) > 100:
                            is_bg_V = True

                    if is_1d_profiles:
                        projection_x = np.asarray(profile_h)
                        projection_x = projection_x - np.mean(projection_x[:50])

                        projection_y = np.asarray(profile_v)
                        projection_y = projection_y - np.mean(projection_y[:50])

                        # Calculate CPC if background image
                        if is_bg_H or is_bg_V:
                            # if 1D profile, img_sum approximation
                            img_sum = np.sum(projection_x[264:644]) + np.sum(projection_y[154:454])
                            if img_sum > 0:
                                current_cpc = (bunch_charge * exposure) / (img_sum * INTENSIFIER_GAIN * in_filter)
                                cpc_per_bunch[delaySlots] = current_cpc
                                global_cpc_fallback = current_cpc # update fallback
                    else:
                        projection_x = np.sum(image_2d[y1:y2, :], axis=0)
                        projection_x = projection_x - np.mean(projection_x[:50])

                        projection_y = np.sum(image_2d[:, x1:x2], axis=1)
                        projection_y = projection_y - np.mean(projection_y[:50])

                        # Calculate CPC if background image
                        if is_bg_H or is_bg_V:
                            img_sum = np.sum(image_2d[y1:y2, 264:644])
                            if img_sum > 0:
                                current_cpc = (bunch_charge * exposure) / (img_sum * INTENSIFIER_GAIN * in_filter)
                                cpc_per_bunch[delaySlots] = current_cpc
                                global_cpc_fallback = current_cpc # update fallback

                    current_cpc = cpc_per_bunch.get(delaySlots, global_cpc_fallback)

                    # Initialize bunch lists if needed
                    for plane_dict in [self.results['H']['bsrh'], self.results['V']['bsrh']]:
                        if delaySlots not in plane_dict:
                            plane_dict[delaySlots] = {
                                'time': [], 'rpi': [], 'is_bg': [],
                                'filepath': [], 'event_idx': [],
                                'maskX': [], 'maskY': [], 'maskR': []
                            }

                    # --- Horizontal ---
                    total_integral_x = np.sum(projection_x)
                    if total_integral_x > 0:
                        if is_bg_H:
                            # 1D Deconvolution
                            x_indices = np.arange(len(projection_x))
                            psf = gaussian(x_indices, 1.0, len(projection_x)//2, psfWidthH)
                            psf = psf / np.sum(psf)
                            deconv_x = richardson_lucy_1d(projection_x, psf, rl_iterations)

                            try:
                                popt, _ = curve_fit(gaussian, x_indices, deconv_x, p0=[np.max(deconv_x), np.argmax(deconv_x), 20.0])
                                cenX = popt[1]
                            except:
                                cenX = np.argmax(deconv_x)

                            xl1 = int(cenX - sigma * SIGMA_TO_MM_X / pixelSizeH)
                            xr1 = int(cenX + sigma * SIGMA_TO_MM_X / pixelSizeH)

                            tails = 0
                            if xl1 > 0: tails += np.sum(deconv_x[:xl1])
                            if xr1 < len(deconv_x): tails += np.sum(deconv_x[xr1:])

                            tails_charge = tails * current_cpc * INTENSIFIER_GAIN * in_filter / exposure
                            rpiH = tails_charge / bunch_charge

                            self.results['H']['bsrh'][delaySlots]['time'].append(timestamp)
                            self.results['H']['bsrh'][delaySlots]['rpi'].append(rpiH)
                            self.results['H']['bsrh'][delaySlots]['is_bg'].append(True)
                            self.results['H']['bsrh'][delaySlots]['filepath'].append(filepath)
                            self.results['H']['bsrh'][delaySlots]['event_idx'].append(event_idx)
                            self.results['H']['bsrh'][delaySlots]['maskX'].append(maskX)
                            self.results['H']['bsrh'][delaySlots]['maskY'].append(maskY)
                            self.results['H']['bsrh'][delaySlots]['maskR'].append(maskR)

                        else:
                            # Masked calculation
                            x_temp = list(range(max(0, maskX - maskR))) + list(range(min(len(projection_x), maskX + maskR), len(projection_x)))
                            valid_x = np.array(x_temp)
                            valid_proj_x = projection_x[valid_x]

                            try:
                                resX, _ = curve_fit(gaussian, valid_x, valid_proj_x, p0=[np.max(valid_proj_x), maskX, 20.0])
                                cenX = resX[1]
                                if abs(cenX - maskX) > maskR:
                                    print(f"Warning: Core outside mask (X). Fit: {cenX}, Mask: {maskX}")
                                    cenX = maskX # fallback
                            except:
                                cenX = maskX

                            xl1 = int(cenX - sigma * SIGMA_TO_MM_X / pixelSizeH)
                            xr1 = int(cenX + sigma * SIGMA_TO_MM_X / pixelSizeH)

                            tails = 0
                            if 0 < xl1 < maskX - maskR:
                                tails += np.sum(projection_x[:xl1])
                            elif xl1 <= 0:
                                pass # no left tail

                            if len(projection_x) > xr1 > maskX + maskR:
                                tails += np.sum(projection_x[xr1:])
                            elif xr1 >= len(projection_x):
                                pass

                            tails_charge = tails * current_cpc * INTENSIFIER_GAIN * in_filter / exposure
                            rpiH = tails_charge / bunch_charge

                            self.results['H']['bsrh'][delaySlots]['time'].append(timestamp)
                            self.results['H']['bsrh'][delaySlots]['rpi'].append(rpiH)
                            self.results['H']['bsrh'][delaySlots]['is_bg'].append(False)
                            self.results['H']['bsrh'][delaySlots]['filepath'].append(filepath)
                            self.results['H']['bsrh'][delaySlots]['event_idx'].append(event_idx)
                            self.results['H']['bsrh'][delaySlots]['maskX'].append(maskX)
                            self.results['H']['bsrh'][delaySlots]['maskY'].append(maskY)
                            self.results['H']['bsrh'][delaySlots]['maskR'].append(maskR)

                    # --- Vertical ---
                    total_integral_y = np.sum(projection_y)
                    if total_integral_y > 0:
                        if is_bg_V:
                            # 1D Deconvolution
                            y_indices = np.arange(len(projection_y))
                            psf = gaussian(y_indices, 1.0, len(projection_y)//2, psfWidthV)
                            psf = psf / np.sum(psf)
                            deconv_y = richardson_lucy_1d(projection_y, psf, rl_iterations)

                            try:
                                popt, _ = curve_fit(gaussian, y_indices, deconv_y, p0=[np.max(deconv_y), np.argmax(deconv_y), 20.0])
                                cenY = popt[1]
                            except:
                                cenY = np.argmax(deconv_y)

                            yl1 = int(cenY - sigma * SIGMA_TO_MM_Y / pixelSizeV)
                            yr1 = int(cenY + sigma * SIGMA_TO_MM_Y / pixelSizeV)

                            tails = 0
                            if yl1 > 0: tails += np.sum(deconv_y[:yl1])
                            if yr1 < len(deconv_y): tails += np.sum(deconv_y[yr1:])

                            tails_charge = tails * current_cpc * INTENSIFIER_GAIN * in_filter / exposure
                            rpiV = tails_charge / bunch_charge

                            self.results['V']['bsrh'][delaySlots]['time'].append(timestamp)
                            self.results['V']['bsrh'][delaySlots]['rpi'].append(rpiV)
                            self.results['V']['bsrh'][delaySlots]['is_bg'].append(True)
                            self.results['V']['bsrh'][delaySlots]['filepath'].append(filepath)
                            self.results['V']['bsrh'][delaySlots]['event_idx'].append(event_idx)
                            self.results['V']['bsrh'][delaySlots]['maskX'].append(maskX)
                            self.results['V']['bsrh'][delaySlots]['maskY'].append(maskY)
                            self.results['V']['bsrh'][delaySlots]['maskR'].append(maskR)

                        else:
                            # Masked calculation
                            y_temp = list(range(max(0, maskY - maskR))) + list(range(min(len(projection_y), maskY + maskR), len(projection_y)))
                            valid_y = np.array(y_temp)
                            valid_proj_y = projection_y[valid_y]

                            try:
                                resY, _ = curve_fit(gaussian, valid_y, valid_proj_y, p0=[np.max(valid_proj_y), maskY, 20.0])
                                cenY = resY[1]
                                if abs(cenY - maskY) > maskR:
                                    print(f"Warning: Core outside mask (Y). Fit: {cenY}, Mask: {maskY}")
                                    cenY = maskY # fallback
                            except:
                                cenY = maskY

                            yl1 = int(cenY - sigma * SIGMA_TO_MM_Y / pixelSizeV)
                            yr1 = int(cenY + sigma * SIGMA_TO_MM_Y / pixelSizeV)

                            tails = 0
                            if 0 < yl1 < maskY - maskR:
                                tails += np.sum(projection_y[:yl1])
                            elif yl1 <= 0:
                                pass # no left tail

                            if len(projection_y) > yr1 > maskY + maskR:
                                tails += np.sum(projection_y[yr1:])
                            elif yr1 >= len(projection_y):
                                pass

                            tails_charge = tails * current_cpc * INTENSIFIER_GAIN * in_filter / exposure
                            rpiV = tails_charge / bunch_charge

                            self.results['V']['bsrh'][delaySlots]['time'].append(timestamp)
                            self.results['V']['bsrh'][delaySlots]['rpi'].append(rpiV)
                            self.results['V']['bsrh'][delaySlots]['is_bg'].append(False)
                            self.results['V']['bsrh'][delaySlots]['filepath'].append(filepath)
                            self.results['V']['bsrh'][delaySlots]['event_idx'].append(event_idx)
                            self.results['V']['bsrh'][delaySlots]['maskX'].append(maskX)
                            self.results['V']['bsrh'][delaySlots]['maskY'].append(maskY)
                            self.results['V']['bsrh'][delaySlots]['maskR'].append(maskR)
                except Exception as e:
                    print(f"Error processing event: {e}")
                    traceback.print_exc()

        self.progress_signal.emit(100)
        self.status_signal.emit("BSRH Analysis Complete")

    def fetch_bws_data(self):
        t1, t2, bunches_to_fetch = self.bws_args

        try:
            ldb = pytimber.LoggingDB(source="nxcals")

            self.status_signal.emit("Fetching BWS via pytimber (Horizontal)...")
            PointNumberH = ldb.get('LHC.BWS.5L4.B2H1:Acquisition:nbPtsPerProjInSet1', t1, t2)
            Pos_inH = ldb.get('LHC.BWS.5L4.B2H1:Acquisition:projPositionSet1', t1, t2)
            Proj_inH = ldb.get('LHC.BWS.5L4.B2H1:Acquisition:projDataSet1', t1, t2)

            self.status_signal.emit("Fetching BWS via pytimber (Vertical)...")
            PointNumberV = ldb.get('LHC.BWS.5R4.B2V1:Acquisition:nbPtsPerProjInSet1', t1, t2)
            Pos_inV = ldb.get('LHC.BWS.5R4.B2V1:Acquisition:projPositionSet1', t1, t2)
            Proj_inV = ldb.get('LHC.BWS.5R4.B2V1:Acquisition:projDataSet1', t1, t2)

            self.status_signal.emit("Processing BWS Data...")

            def process_bws(PointNumber, Pos_in, Proj_in, plane, size_mm):
                if not PointNumber or not Pos_in or not Proj_in: return
                _, p_num = list(PointNumber.items())[0]
                _, p_pos = list(Pos_in.items())[0]
                _, p_proj = list(Proj_in.items())[0]

                # timestamps are p_pos[0]
                timestamps = p_pos[0]

                for bunch in bunches_to_fetch:
                    if bunch not in self.results[plane]['bws']:
                        self.results[plane]['bws'][bunch] = {'time': [], 'rpi': []}

                    for j in range(len(timestamps)):
                        if not self.is_running:
                            return
                        try:
                            scan_time = dt.datetime.fromtimestamp(timestamps[j])
                            count = int(p_num[1][j])

                            x = p_pos[1][j][0][:count]

                            start_idx = bunch * count
                            end_idx = (bunch + 1) * count
                            if end_idx > len(p_proj[1][j][0]):
                                continue # Bunch index out of range

                            y = p_proj[1][j][0][start_idx:end_idx]
                            y = y - np.mean(y[:50])

                            total_integral = np.sum(y)
                            if total_integral <= 0: continue

                            # Fit to find center and size
                            popt, _ = curve_fit(gaussian, x, y, p0=[np.max(y), np.mean(x), 0.28])
                            cen = popt[1]

                            # calculate RPI
                            left_edge = cen - self.params['sigma'] * size_mm
                            right_edge = cen + self.params['sigma'] * size_mm

                            tails = 0
                            for k in range(len(x)):
                                if x[k] < left_edge or x[k] > right_edge:
                                    tails += y[k]

                            rpi = tails / total_integral
                            self.results[plane]['bws'][bunch]['time'].append(scan_time)
                            self.results[plane]['bws'][bunch]['rpi'].append(rpi)
                        except Exception as e:
                            print(f"BWS Process Error Event {j}: {e}")

            hSize = 257.0 / 1000.0
            vSize = 386.0 / 1000.0

            process_bws(PointNumberH, Pos_inH, Proj_inH, 'H', hSize)
            process_bws(PointNumberV, Pos_inV, Proj_inV, 'V', vSize)

            self.status_signal.emit("BWS Fetch Complete")
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(f"Failed to fetch data: {e}")

def richardson_lucy_1d(image, psf, iterations=10):
    """
    1D Richardson-Lucy deconvolution.
    image: 1D numpy array (the blurred signal)
    psf: 1D numpy array (the point spread function, should sum to 1)
    iterations: number of iterations
    """
    # Ensure PSF is normalized
    psf = psf / np.sum(psf)
    # The mirrored PSF for correlation
    psf_mirror = psf[::-1]

    # Initialize the estimate with the original image (avoid zeros)
    estimate = np.copy(image)
    estimate[estimate <= 0] = 1e-6

    for _ in range(iterations):
        # Convolve current estimate with PSF
        conv = convolve(estimate, psf, mode='same')
        conv[conv <= 0] = 1e-6

        # Calculate relative blur
        relative_blur = image / conv

        # Correlate relative blur with mirrored PSF
        error_est = convolve(relative_blur, psf_mirror, mode='same')

        # Update estimate
        estimate = estimate * error_est

    return estimate

class AnalysisOfflineGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offline BSRH/BWS RPI Analysis GUI")
        self.resize(1200, 800)

        # Main layout
        main_layout = QHBoxLayout(self)

        # Splitter for resizable left/right panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # LEFT PANEL (Controls & Parameters)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)

        # 1. Directory Selection Group
        dir_group = QGroupBox("Data Directories")
        dir_layout = QVBoxLayout()

        self.dir_scroll = QScrollArea()
        self.dir_scroll.setWidgetResizable(True)
        self.dir_container = QWidget()
        self.dir_container_layout = QVBoxLayout(self.dir_container)
        self.dir_container_layout.setAlignment(Qt.AlignTop)
        self.dir_scroll.setWidget(self.dir_container)
        dir_layout.addWidget(self.dir_scroll)

        btn_layout = QHBoxLayout()
        self.btn_add_dir = QPushButton("Add Dir")
        self.btn_add_dir.clicked.connect(self.add_directory)
        btn_layout.addWidget(self.btn_add_dir)
        dir_layout.addLayout(btn_layout)
        dir_group.setLayout(dir_layout)
        left_layout.addWidget(dir_group)

        self.dir_widgets = []

        # 2. Parameters Group
        param_group = QGroupBox("Analysis Parameters")
        param_layout = QFormLayout()

        self.param_inputs = {}

        # Default values based on user feedback
        params = {
            "sigma": "4.7",
            "pixelSizeH": "21.4",
            "pixelSizeV": "18.0",
            "psfWidthH": "450", # PSF width in pixels for deconvolution
            "psfWidthV": "270",
            "rl_iterations": "5"
        }

        for p_name, p_val in params.items():
            line_edit = QLineEdit(p_val)
            self.param_inputs[p_name] = line_edit
            param_layout.addRow(QLabel(p_name + ":"), line_edit)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # 3. Execution & View Controls Group
        exec_group = QGroupBox("Execution & View")
        exec_layout = QVBoxLayout()

        view_layout = QHBoxLayout()

        bunch_layout = QVBoxLayout()
        bunch_layout.addWidget(QLabel("Select Bunches:"))
        self.bunch_list = QListWidget()
        self.bunch_list.setSelectionMode(QListWidget.MultiSelection)
        self.bunch_list.itemSelectionChanged.connect(self.update_plot)
        bunch_layout.addWidget(self.bunch_list)

        view_layout.addLayout(bunch_layout)

        plane_layout = QVBoxLayout()
        plane_layout.addWidget(QLabel("Select Plane:"))
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["Horizontal", "Vertical"])
        self.plane_combo.currentIndexChanged.connect(self.update_plot)
        plane_layout.addWidget(self.plane_combo)
        plane_layout.addStretch()

        view_layout.addLayout(plane_layout)
        exec_layout.addLayout(view_layout)

        btn_action_layout = QHBoxLayout()
        self.btn_refine_mask = QPushButton("Refine Mask Position")
        self.btn_refine_mask.clicked.connect(self.refine_mask_position)
        btn_action_layout.addWidget(self.btn_refine_mask)

        self.btn_start_bsrh = QPushButton("Start BSRH Analysis")
        self.btn_start_bsrh.clicked.connect(self.start_bsrh_analysis)
        btn_action_layout.addWidget(self.btn_start_bsrh)

        exec_layout.addLayout(btn_action_layout)
        self.btn_start_bsrh.clicked.connect(self.start_bsrh_analysis)
        exec_layout.addWidget(self.btn_start_bsrh)

        self.btn_fetch_bws = QPushButton("Fetch BWS via Pytimber")
        self.btn_fetch_bws.clicked.connect(self.fetch_bws_data)
        if not HAS_PYTIMBER:
            self.btn_fetch_bws.setEnabled(False)
            self.btn_fetch_bws.setToolTip("pytimber not installed")
        exec_layout.addWidget(self.btn_fetch_bws)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        exec_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready")
        exec_layout.addWidget(self.status_lbl)

        exec_group.setLayout(exec_layout)
        left_layout.addWidget(exec_group)

        left_layout.addStretch()

        # RIGHT PANEL (Plots)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("RPI (Relative Projection Integral)")
        self.ax.set_title("RPI vs Time")
        self.canvas.draw()

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # Set splitter sizes (approx 1/3 and 2/3)
        splitter.setSizes([400, 800])

        # Data structures to hold results
        self.results = {
            'H': {'bsrh': {}, 'bws': {}},
            'V': {'bsrh': {}, 'bws': {}}
        }

        self.worker = None
        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.current_mask = {'X': 455, 'Y': 300, 'R': 40}

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.status_lbl.setText("Stopping worker thread...")
            self.worker.stop()
            self.worker.wait(2000) # wait up to 2 seconds for clean exit
        event.accept()

    def add_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select BSRH Directory")
        if not dir_path:
            return

        # Count files in the directory
        files = [f for f in os.listdir(dir_path) if f.endswith('.pkl.zst')]
        total_files = len(files)

        if total_files == 0:
            QMessageBox.information(self, "No Files", f"No .pkl.zst files found in:\n{dir_path}")
            return

        widget = DirectoryEntryWidget(dir_path, total_files)
        widget.remove_btn.clicked.connect(lambda: self.remove_directory(widget))
        self.dir_container_layout.addWidget(widget)
        self.dir_widgets.append(widget)

    def remove_directory(self, widget):
        self.dir_container_layout.removeWidget(widget)
        widget.deleteLater()
        if widget in self.dir_widgets:
            self.dir_widgets.remove(widget)

    def get_params(self):
        try:
            return {k: float(v.text()) for k, v in self.param_inputs.items()}
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please ensure all parameters are valid numbers.")
            return None

    def refine_mask_position(self):
        if len(self.dir_widgets) == 0:
            QMessageBox.warning(self, "No Directories", "Please add at least one directory to search for mask images.")
            return

        directories = [widget.get_range() for widget in self.dir_widgets]

        dctx = zstd.ZstdDecompressor()
        all_files = []
        for d, start_idx, end_idx in directories:
            if os.path.isdir(d):
                files = [f for f in os.listdir(d) if f.endswith('.pkl.zst')]
                files = sorted(files)
                end_idx = min(end_idx, len(files) - 1)
                if start_idx <= end_idx and start_idx < len(files):
                    files_in_range = files[start_idx:end_idx+1]
                    all_files.extend([os.path.join(d, f) for f in files_in_range])

        found_masked = False
        maskX, maskY, maskR = self.current_mask['X'], self.current_mask['Y'], self.current_mask['R']

        for filepath in all_files:
            if found_masked: break
            try:
                with open(filepath, 'rb') as f:
                    with dctx.stream_reader(f) as reader:
                        file_data = pickle.load(reader)
                        for event_idx, event in enumerate(file_data):
                            cmask_x = event.device_data.get('CMASK-X', None)
                            if cmask_x is not None:
                                cmask_x_val = np.asarray(cmask_x).item() if isinstance(cmask_x, np.ndarray) else cmask_x
                                if abs(cmask_x_val - self.current_mask['X']) < 150: # broadened threshold slightly

                                    img_data = event.cam_data.get('image2D') if isinstance(event.cam_data, dict) else event.cam_data
                                    # Need a full 2D image for validation overlay
                                    if isinstance(img_data, list) or (hasattr(img_data, 'shape') and img_data.shape[0] <= 2):
                                        continue # Keep searching for a true 2D image

                                    image_2d = img_data

                                    # Auto-guess ONLY if we haven't touched defaults yet, else use current
                                    if self.current_mask['X'] == 455 and self.current_mask['Y'] == 300:
                                        center_region = image_2d[154:454, 284:684]
                                        min_idx = np.unravel_index(np.argmin(center_region, axis=None), center_region.shape)
                                        maskY = min_idx[0] + 154
                                        maskX = min_idx[1] + 284

                                    dialog = MaskValidationDialog(image_2d, maskX, maskY, maskR, self)
                                    if dialog.exec_() == QDialog.Accepted:
                                        nX, nY, nR = dialog.get_mask_params()
                                        self.current_mask['X'] = nX
                                        self.current_mask['Y'] = nY
                                        self.current_mask['R'] = nR
                                        QMessageBox.information(self, "Mask Validated", f"Mask position updated to: X={nX}, Y={nY}, R={nR}")
                                    else:
                                        self.status_lbl.setText("Mask refinement cancelled.")

                                    found_masked = True
                                    break
            except Exception as e:
                pass

        if not found_masked:
            QMessageBox.warning(self, "Warning", "Could not find a 2D masked image to validate the mask.")

    def start_bsrh_analysis(self):
        if len(self.dir_widgets) == 0:
            QMessageBox.warning(self, "No Directories", "Please add at least one directory.")
            return

        params = self.get_params()
        if not params:
            return

        directories = [widget.get_range() for widget in self.dir_widgets]

        params['maskX'] = self.current_mask['X']
        params['maskY'] = self.current_mask['Y']
        params['maskR'] = self.current_mask['R']

        self.btn_start_bsrh.setEnabled(False)
        self.btn_fetch_bws.setEnabled(False)
        self.btn_refine_mask.setEnabled(False)
        self.status_lbl.setText("Analyzing BSRH...")

        # Stop existing worker if any (shouldn't happen with buttons disabled, but safe)
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        self.worker = AnalysisWorker('BSRH', directories=directories, params=params, parent=self)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.status_signal.connect(self.status_lbl.setText)
        self.worker.error_signal.connect(self.handle_worker_error)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.start()

    def fetch_bws_data(self):
        if not HAS_PYTIMBER:
            QMessageBox.warning(self, "Pytimber Missing", "Pytimber module is not installed.")
            return

        params = self.get_params()
        if not params:
            return

        # Ask for start and end time
        dialog = QDialog(self)
        dialog.setWindowTitle("Fetch BWS Data")
        layout = QFormLayout(dialog)

        start_input = QLineEdit("2024-03-13 12:00:00")
        end_input = QLineEdit("2024-03-14 08:50:00")
        bunch_input = QLineEdit("1") # The pytimber snippet used '1' as a bunch offset

        layout.addRow("Start Time (YYYY-MM-DD HH:MM:SS):", start_input)
        layout.addRow("End Time (YYYY-MM-DD HH:MM:SS):", end_input)
        layout.addRow("Bunches (comma separated):", bunch_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            t1 = dt.datetime.strptime(start_input.text(), "%Y-%m-%d %H:%M:%S")
            t2 = dt.datetime.strptime(end_input.text(), "%Y-%m-%d %H:%M:%S")
            bunches_to_fetch = [int(b.strip()) for b in bunch_input.text().split(',')]
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid date format or bunch list.")
            return

        try:
            self.btn_start_bsrh.setEnabled(False)
            self.btn_fetch_bws.setEnabled(False)

            if self.worker and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait()

            self.worker = AnalysisWorker('BWS', bws_args=(t1, t2, bunches_to_fetch), params=params, parent=self)
            self.worker.progress_signal.connect(self.progress_bar.setValue)
            self.worker.status_signal.connect(self.status_lbl.setText)
            self.worker.error_signal.connect(self.handle_worker_error)
            self.worker.finished_signal.connect(self.analysis_finished)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Pytimber Error", f"Failed to fetch data: {e}")
            self.status_lbl.setText("BWS Fetch Failed")

    def handle_worker_error(self, err_str):
        QMessageBox.critical(self, "Analysis Error", err_str)
        self.status_lbl.setText("Error occurred.")
        self.btn_start_bsrh.setEnabled(True)
        self.btn_refine_mask.setEnabled(True)
        if HAS_PYTIMBER: self.btn_fetch_bws.setEnabled(True)

    def analysis_finished(self, new_results):
        # Update self.results with the new data
        for plane in ['H', 'V']:
            # For BSRH
            for bunch, data in new_results[plane]['bsrh'].items():
                if bunch not in self.results[plane]['bsrh']:
                    self.results[plane]['bsrh'][bunch] = {
                        'time': [], 'rpi': [], 'is_bg': [],
                        'filepath': [], 'event_idx': [],
                        'maskX': [], 'maskY': [], 'maskR': []
                    }
                self.results[plane]['bsrh'][bunch]['time'].extend(data['time'])
                self.results[plane]['bsrh'][bunch]['rpi'].extend(data['rpi'])
                self.results[plane]['bsrh'][bunch]['is_bg'].extend(data['is_bg'])
                self.results[plane]['bsrh'][bunch]['filepath'].extend(data.get('filepath', []))
                self.results[plane]['bsrh'][bunch]['event_idx'].extend(data.get('event_idx', []))
                self.results[plane]['bsrh'][bunch]['maskX'].extend(data.get('maskX', []))
                self.results[plane]['bsrh'][bunch]['maskY'].extend(data.get('maskY', []))
                self.results[plane]['bsrh'][bunch]['maskR'].extend(data.get('maskR', []))
            # For BWS
            for bunch, data in new_results[plane]['bws'].items():
                if bunch not in self.results[plane]['bws']:
                    self.results[plane]['bws'][bunch] = {'time': [], 'rpi': []}
                self.results[plane]['bws'][bunch]['time'].extend(data['time'])
                self.results[plane]['bws'][bunch]['rpi'].extend(data['rpi'])

        self.btn_start_bsrh.setEnabled(True)
        self.btn_refine_mask.setEnabled(True)
        if HAS_PYTIMBER: self.btn_fetch_bws.setEnabled(True)
        self.update_bunch_list()
        self.update_plot()

    def run_bsrh_analysis(self, directories, params):
        self.results['H']['bsrh'].clear()
        self.results['V']['bsrh'].clear()

        sigma = params['sigma']
        pixelSizeH = params['pixelSizeH']
        pixelSizeV = params['pixelSizeV']
        psfWidthH = params['psfWidthH']
        psfWidthV = params['psfWidthV']
        rl_iterations = int(params['rl_iterations'])

        # Hardcoded Constants
        INTENSIFIER_GAIN = 5651.0

        SIGMA_TO_MM_X = 263.0
        SIGMA_TO_MM_Y = 361.0

        y1, y2 = 154, 454
        x1, x2 = 284, 684

        dctx = zstd.ZstdDecompressor()

        total_files = 0

        # Track cPc per bunch dynamically from unmasked background images
        cpc_per_bunch = {}
        # Fallback if an unmasked image hasn't been seen for a bunch yet
        global_cpc_fallback = 44.0
        all_files = []
        for d, start_idx, end_idx in directories:
            if os.path.isdir(d):
                files = [f for f in os.listdir(d) if f.endswith('.pkl.zst')]
                files = sorted(files)
                # Apply range
                end_idx = min(end_idx, len(files) - 1)
                if start_idx <= end_idx and start_idx < len(files):
                    files_in_range = files[start_idx:end_idx+1]
                    all_files.extend([os.path.join(d, f) for f in files_in_range])

        total_files = len(all_files)

        if total_files == 0:
            QMessageBox.information(self, "No files", "No .pkl.zst files found in selected directories/ranges.")
            return

        # 1. Find the first masked image to auto-detect and validate mask
        maskX, maskY, maskR = 455, 300, 40 # defaults
        found_masked = False

        for filepath in all_files:
            if found_masked: break
            try:
                with open(filepath, 'rb') as f:
                    with dctx.stream_reader(f) as reader:
                        file_data = pickle.load(reader)
                        for event in file_data:
                            # Assume it's a masked image if CMASK is present and not explicitly "background"
                            cmask_x = event.device_data.get('CMASK-X', None)
                            if cmask_x is not None:
                                cmask_x_val = np.asarray(cmask_x).item() if isinstance(cmask_x, np.ndarray) else cmask_x
                                if abs(cmask_x_val - 455) < 100:
                                    image = event.cam_data['image2D']
                                    if image.shape[0] <= 2:
                                        # It's a 1D profile, we can't do the 2D mask validation easily. Use defaults.
                                        maskX, maskY, maskR = 455, 300, 40
                                        found_masked = True
                                        break

                                    # Simple auto-guess: find minimum in center region
                                    center_region = image[154:454, 284:684]
                                    # find the index of the minimum value in the 2D array
                                    min_idx = np.unravel_index(np.argmin(center_region, axis=None), center_region.shape)
                                    # min_idx is (y, x) within the region
                                    guessedY = min_idx[0] + 154
                                    guessedX = min_idx[1] + 284

                                    # Show dialog to user
                                    dialog = MaskValidationDialog(image_2d, guessedX, guessedY, maskR, self)
                                    if dialog.exec_() == QDialog.Accepted:
                                        maskX, maskY, maskR = dialog.get_mask_params()
                                        found_masked = True
                                    else:
                                        self.status_lbl.setText("Analysis aborted (mask validation cancelled).")
                                        return
                                    break
            except Exception as e:
                print(f"Error checking {filepath} for mask: {e}")

        if not found_masked:
            QMessageBox.warning(self, "Warning", "Could not find a masked image to validate the mask. Using defaults.")
            maskX, maskY, maskR = 455, 300, 40

        for i, filepath in enumerate(all_files):
            self.progress_bar.setValue(int((i / total_files) * 100))
            QApplication.processEvents()

            try:
                with open(filepath, 'rb') as f:
                    with dctx.stream_reader(f) as reader:
                        file_data = pickle.load(reader)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            for event_idx, event in enumerate(file_data):
                try:
                    timestamp = event.timestamp
                    if timestamp.tzinfo is not None:
                        timestamp = timestamp.replace(tzinfo=None)

                    delaySlots_raw = event.device_data.get("GATE-INT", {}).get("delaySlots", -1)
                    delaySlots = int(np.asarray(delaySlots_raw).item()) if isinstance(delaySlots_raw, np.ndarray) else int(delaySlots_raw)
                    if delaySlots == -1:
                        continue

                    tmpFBCT = event.device_data.get('FBCT1', {}).get('averageBunchIntensities', [])
                    if isinstance(tmpFBCT, np.ndarray) and tmpFBCT.size == 0:
                        tmpFBCT = []
                    bunch_charge = 0
                    if len(tmpFBCT) > delaySlots:
                        # Attempt to handle pulseLength if present (like in cpc_bunches.py)
                        pulse_length_raw = event.device_data.get("GATE-INT", {}).get("pulseLength", 4)
                        pulse_length = int(np.asarray(pulse_length_raw).item()) if isinstance(pulse_length_raw, np.ndarray) else int(pulse_length_raw)

                        end_slot = min(len(tmpFBCT), delaySlots + max(1, pulse_length // 4))
                        bunch_charge = np.sum(tmpFBCT[delaySlots:end_slot])

                    # Fix truth value of an array
                    if isinstance(bunch_charge, np.ndarray):
                        if bunch_charge.size == 1:
                            bunch_charge = bunch_charge.item()
                        else:
                            bunch_charge = np.sum(bunch_charge)

                    if bunch_charge <= 0:
                        continue # Can't calculate relative projection without a reference FBCT charge

                    image = event.cam_data['image2D']
                    exposure_raw = event.device_data.get('GATE-INT', {}).get('numberOfPulses', 1)
                    exposure = int(np.asarray(exposure_raw).item()) if isinstance(exposure_raw, np.ndarray) else int(exposure_raw)
                    if exposure == 0: exposure = 1

                    fArr = event.device_data.get('FILTERS.B2', {}).get('channelState', [0,0,0])
                    # Ensure scalars for the math
                    fArr0 = np.asarray(fArr[0]).item() if isinstance(fArr[0], np.ndarray) else fArr[0]
                    fArr1 = np.asarray(fArr[1]).item() if isinstance(fArr[1], np.ndarray) else fArr[1]
                    fArr2 = np.asarray(fArr[2]).item() if isinstance(fArr[2], np.ndarray) else fArr[2]
                    in_filter = pow(12.6, 3 * fArr0 + 2 * fArr1 + fArr2)

                    cmask_x = event.device_data.get('CMASK-X', None)
                    cmask_y = event.device_data.get('CMASK-Y', None)

                    # Assuming mask is present if CMASK is close to mask position
                    # "different CMASK-X or Y value" for background
                    # Let's say if abs(cmask_x - maskX) < some_threshold, it's masked
                    is_bg_H = False
                    if cmask_x is not None:
                        cmask_x_val = np.asarray(cmask_x).item() if isinstance(cmask_x, np.ndarray) else cmask_x
                        if abs(cmask_x_val - maskX) > 100:
                            is_bg_H = True
                    # If there's no CMASK info, assume masked for safety

                    is_bg_V = False
                    if cmask_y is not None:
                        cmask_y_val = np.asarray(cmask_y).item() if isinstance(cmask_y, np.ndarray) else cmask_y
                        if abs(cmask_y_val - maskY) > 100:
                            is_bg_V = True

                    if image.shape[0] <= 2:
                        projection_x = image[0]
                        projection_x = projection_x - np.mean(projection_x[:50])
                        if image.shape[0] > 1:
                            projection_y = image[1]
                        else:
                            projection_y = np.zeros_like(projection_x)
                        projection_y = projection_y - np.mean(projection_y[:50])

                        # Calculate CPC if background image
                        if is_bg_H or is_bg_V:
                            # if 1D profile, img_sum approximation
                            img_sum = np.sum(projection_x[264:644]) + np.sum(projection_y[154:454])
                            if img_sum > 0:
                                current_cpc = (bunch_charge * exposure) / (img_sum * INTENSIFIER_GAIN * in_filter)
                                cpc_per_bunch[delaySlots] = current_cpc
                                global_cpc_fallback = current_cpc # update fallback
                    else:
                        projection_x = np.sum(image[y1:y2, :], axis=0)
                        projection_x = projection_x - np.mean(projection_x[:50])

                        projection_y = np.sum(image[:, x1:x2], axis=1)
                        projection_y = projection_y - np.mean(projection_y[:50])

                        # Calculate CPC if background image
                        if is_bg_H or is_bg_V:
                            img_sum = np.sum(image[y1:y2, 264:644])
                            if img_sum > 0:
                                current_cpc = (bunch_charge * exposure) / (img_sum * INTENSIFIER_GAIN * in_filter)
                                cpc_per_bunch[delaySlots] = current_cpc
                                global_cpc_fallback = current_cpc # update fallback

                    current_cpc = cpc_per_bunch.get(delaySlots, global_cpc_fallback)

                    # Initialize bunch lists if needed
                    for plane_dict in [self.results['H']['bsrh'], self.results['V']['bsrh']]:
                        if delaySlots not in plane_dict:
                            plane_dict[delaySlots] = {'time': [], 'rpi': [], 'is_bg': []}

                    # --- Horizontal ---
                    total_integral_x = np.sum(projection_x)
                    if total_integral_x > 0:
                        if is_bg_H:
                            # 1D Deconvolution
                            x_indices = np.arange(len(projection_x))
                            psf = gaussian(x_indices, 1.0, len(projection_x)//2, params['psfWidthH'])
                            psf = psf / np.sum(psf)
                            deconv_x = richardson_lucy_1d(projection_x, psf, rl_iterations)

                            try:
                                popt, _ = curve_fit(gaussian, x_indices, deconv_x, p0=[np.max(deconv_x), np.argmax(deconv_x), 20.0])
                                cenX = popt[1]
                            except:
                                cenX = np.argmax(deconv_x)

                            xl1 = int(cenX - sigma * SIGMA_TO_MM_X / pixelSizeH)
                            xr1 = int(cenX + sigma * SIGMA_TO_MM_X / pixelSizeH)

                            tails = 0
                            if xl1 > 0: tails += np.sum(deconv_x[:xl1])
                            if xr1 < len(deconv_x): tails += np.sum(deconv_x[xr1:])

                            tails_charge = tails * current_cpc * INTENSIFIER_GAIN * in_filter / exposure
                            rpiH = tails_charge / bunch_charge

                            self.results['H']['bsrh'][delaySlots]['time'].append(timestamp)
                            self.results['H']['bsrh'][delaySlots]['rpi'].append(rpiH)
                            self.results['H']['bsrh'][delaySlots]['is_bg'].append(True)

                        else:
                            # Masked calculation
                            x_temp = list(range(max(0, maskX - maskR))) + list(range(min(len(projection_x), maskX + maskR), len(projection_x)))
                            valid_x = np.array(x_temp)
                            valid_proj_x = projection_x[valid_x]

                            try:
                                resX, _ = curve_fit(gaussian, valid_x, valid_proj_x, p0=[np.max(valid_proj_x), maskX, 20.0])
                                cenX = resX[1]
                                if abs(cenX - maskX) > maskR:
                                    print(f"Warning: Core outside mask (X). Fit: {cenX}, Mask: {maskX}")
                                    cenX = maskX # fallback
                            except:
                                cenX = maskX

                            xl1 = int(cenX - sigma * SIGMA_TO_MM_X / pixelSizeH)
                            xr1 = int(cenX + sigma * SIGMA_TO_MM_X / pixelSizeH)

                            tails = 0
                            if 0 < xl1 < maskX - maskR:
                                tails += np.sum(projection_x[:xl1])
                            elif xl1 <= 0:
                                pass # no left tail

                            if len(projection_x) > xr1 > maskX + maskR:
                                tails += np.sum(projection_x[xr1:])
                            elif xr1 >= len(projection_x):
                                pass

                            tails_charge = tails * current_cpc * INTENSIFIER_GAIN * in_filter / exposure
                            rpiH = tails_charge / bunch_charge

                            self.results['H']['bsrh'][delaySlots]['time'].append(timestamp)
                            self.results['H']['bsrh'][delaySlots]['rpi'].append(rpiH)
                            self.results['H']['bsrh'][delaySlots]['is_bg'].append(False)

                    # --- Vertical ---
                    total_integral_y = np.sum(projection_y)
                    if total_integral_y > 0:
                        if is_bg_V:
                            # 1D Deconvolution
                            y_indices = np.arange(len(projection_y))
                            psf = gaussian(y_indices, 1.0, len(projection_y)//2, params['psfWidthV'])
                            psf = psf / np.sum(psf)
                            deconv_y = richardson_lucy_1d(projection_y, psf, rl_iterations)

                            try:
                                popt, _ = curve_fit(gaussian, y_indices, deconv_y, p0=[np.max(deconv_y), np.argmax(deconv_y), 20.0])
                                cenY = popt[1]
                            except:
                                cenY = np.argmax(deconv_y)

                            yl1 = int(cenY - sigma * SIGMA_TO_MM_Y / pixelSizeV)
                            yr1 = int(cenY + sigma * SIGMA_TO_MM_Y / pixelSizeV)

                            tails = 0
                            if yl1 > 0: tails += np.sum(deconv_y[:yl1])
                            if yr1 < len(deconv_y): tails += np.sum(deconv_y[yr1:])

                            tails_charge = tails * current_cpc * INTENSIFIER_GAIN * in_filter / exposure
                            rpiV = tails_charge / bunch_charge

                            self.results['V']['bsrh'][delaySlots]['time'].append(timestamp)
                            self.results['V']['bsrh'][delaySlots]['rpi'].append(rpiV)
                            self.results['V']['bsrh'][delaySlots]['is_bg'].append(True)

                        else:
                            # Masked calculation
                            y_temp = list(range(max(0, maskY - maskR))) + list(range(min(len(projection_y), maskY + maskR), len(projection_y)))
                            valid_y = np.array(y_temp)
                            valid_proj_y = projection_y[valid_y]

                            try:
                                resY, _ = curve_fit(gaussian, valid_y, valid_proj_y, p0=[np.max(valid_proj_y), maskY, 20.0])
                                cenY = resY[1]
                                if abs(cenY - maskY) > maskR:
                                    print(f"Warning: Core outside mask (Y). Fit: {cenY}, Mask: {maskY}")
                                    cenY = maskY # fallback
                            except:
                                cenY = maskY

                            yl1 = int(cenY - sigma * SIGMA_TO_MM_Y / pixelSizeV)
                            yr1 = int(cenY + sigma * SIGMA_TO_MM_Y / pixelSizeV)

                            tails = 0
                            if 0 < yl1 < maskY - maskR:
                                tails += np.sum(projection_y[:yl1])
                            elif yl1 <= 0:
                                pass # no left tail

                            if len(projection_y) > yr1 > maskY + maskR:
                                tails += np.sum(projection_y[yr1:])
                            elif yr1 >= len(projection_y):
                                pass

                            tails_charge = tails * current_cpc * INTENSIFIER_GAIN * in_filter / exposure
                            rpiV = tails_charge / bunch_charge

                            self.results['V']['bsrh'][delaySlots]['time'].append(timestamp)
                            self.results['V']['bsrh'][delaySlots]['rpi'].append(rpiV)
                            self.results['V']['bsrh'][delaySlots]['is_bg'].append(False)
                except Exception as e:
                    print(f"Error processing event: {e}")
                    traceback.print_exc()

        self.progress_bar.setValue(100)
        self.update_bunch_list()

    def fetch_bws_data(self):
        if not HAS_PYTIMBER:
            QMessageBox.warning(self, "Pytimber Missing", "Pytimber module is not installed.")
            return

        params = self.get_params()
        if not params:
            return

        # Ask for start and end time
        dialog = QDialog(self)
        dialog.setWindowTitle("Fetch BWS Data")
        layout = QFormLayout(dialog)

        start_input = QLineEdit("2024-03-13 12:00:00")
        end_input = QLineEdit("2024-03-14 08:50:00")
        bunch_input = QLineEdit("1") # The pytimber snippet used '1' as a bunch offset

        layout.addRow("Start Time (YYYY-MM-DD HH:MM:SS):", start_input)
        layout.addRow("End Time (YYYY-MM-DD HH:MM:SS):", end_input)
        layout.addRow("Bunches (comma separated):", bunch_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            t1 = dt.datetime.strptime(start_input.text(), "%Y-%m-%d %H:%M:%S")
            t2 = dt.datetime.strptime(end_input.text(), "%Y-%m-%d %H:%M:%S")
            bunches_to_fetch = [int(b.strip()) for b in bunch_input.text().split(',')]
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid date format or bunch list.")
            return

        self.status_lbl.setText("Fetching BWS via pytimber...")
        QApplication.processEvents()

        try:
            ldb = pytimber.LoggingDB(source="nxcals")

            # Fetch Horizontal
            PointNumberH = ldb.get('LHC.BWS.5L4.B2H1:Acquisition:nbPtsPerProjInSet1', t1, t2)
            Pos_inH = ldb.get('LHC.BWS.5L4.B2H1:Acquisition:projPositionSet1', t1, t2)
            Proj_inH = ldb.get('LHC.BWS.5L4.B2H1:Acquisition:projDataSet1', t1, t2)

            # Fetch Vertical
            PointNumberV = ldb.get('LHC.BWS.5R4.B2V1:Acquisition:nbPtsPerProjInSet1', t1, t2)
            Pos_inV = ldb.get('LHC.BWS.5R4.B2V1:Acquisition:projPositionSet1', t1, t2)
            Proj_inV = ldb.get('LHC.BWS.5R4.B2V1:Acquisition:projDataSet1', t1, t2)

            # Process function
            def process_bws(PointNumber, Pos_in, Proj_in, plane, size_mm):
                if not PointNumber or not Pos_in or not Proj_in: return
                _, p_num = list(PointNumber.items())[0]
                _, p_pos = list(Pos_in.items())[0]
                _, p_proj = list(Proj_in.items())[0]

                # timestamps are p_pos[0]
                timestamps = p_pos[0]

                for bunch in bunches_to_fetch:
                    if bunch not in self.results[plane]['bws']:
                        self.results[plane]['bws'][bunch] = {'time': [], 'rpi': []}

                    for j in range(len(timestamps)):
                        try:
                            scan_time = dt.datetime.fromtimestamp(timestamps[j])
                            count = int(p_num[1][j])

                            x = p_pos[1][j][0][:count]

                            start_idx = bunch * count
                            end_idx = (bunch + 1) * count
                            if end_idx > len(p_proj[1][j][0]):
                                continue # Bunch index out of range

                            y = p_proj[1][j][0][start_idx:end_idx]
                            y = y - np.mean(y[:50])

                            total_integral = np.sum(y)
                            if total_integral <= 0: continue

                            # Fit to find center and size
                            popt, _ = curve_fit(gaussian, x, y, p0=[np.max(y), np.mean(x), 0.28])
                            cen = popt[1]

                            # calculate RPI
                            # highLeftPosition = cen - sigma * size
                            left_edge = cen - params['sigma'] * size_mm
                            right_edge = cen + params['sigma'] * size_mm

                            tails = 0
                            for k in range(len(x)):
                                if x[k] < left_edge or x[k] > right_edge:
                                    tails += y[k]

                            rpi = tails / total_integral
                            self.results[plane]['bws'][bunch]['time'].append(scan_time)
                            self.results[plane]['bws'][bunch]['rpi'].append(rpi)
                        except Exception as e:
                            print(f"BWS Process Error Event {j}: {e}")

            # Horizontal size approx (e.g., from Offline_Sofia.py)
            hSize = 257.0 / 1000.0
            vSize = 386.0 / 1000.0

            process_bws(PointNumberH, Pos_inH, Proj_inH, 'H', hSize)
            process_bws(PointNumberV, Pos_inV, Proj_inV, 'V', vSize)

            self.status_lbl.setText("BWS Fetch Complete")
        except Exception as e:
            QMessageBox.critical(self, "Pytimber Error", f"Failed to fetch data: {e}")
            self.status_lbl.setText("BWS Fetch Failed")

        self.update_bunch_list()
        self.update_plot()

    def update_bunch_list(self):
        # Gather all unique bunch numbers from results
        all_bunches = set()
        for plane in ['H', 'V']:
            all_bunches.update(self.results[plane]['bsrh'].keys())
            all_bunches.update(self.results[plane]['bws'].keys())

        # Store currently selected items to restore selection
        selected = [item.text() for item in self.bunch_list.selectedItems()]

        self.bunch_list.clear()
        for b in sorted(list(all_bunches)):
            item = QListWidgetItem(str(b))
            self.bunch_list.addItem(item)
            if str(b) in selected:
                item.setSelected(True)

    def update_plot(self):
        self.ax.clear()
        plane = "H" if self.plane_combo.currentText() == "Horizontal" else "V"

        # Read bunches from the list widget
        bunches = [int(item.text()) for item in self.bunch_list.selectedItems()]

        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("RPI")
        self.ax.set_title(f"RPI vs Time ({self.plane_combo.currentText()})")

        # Plot BSRH
        bsrh_data = self.results[plane]['bsrh']
        for b in bunches:
            if b in bsrh_data:
                times = bsrh_data[b]['time']
                rpis = bsrh_data[b]['rpi']
                is_bg = bsrh_data[b]['is_bg']

                # Separate masked and background
                t_masked = [t for i, t in enumerate(times) if not is_bg[i]]
                r_masked = [r for i, r in enumerate(rpis) if not is_bg[i]]

                t_bg = [t for i, t in enumerate(times) if is_bg[i]]
                r_bg = [r for i, r in enumerate(rpis) if is_bg[i]]

                # Keep track of original indices to map back clicks
                idx_masked = [i for i, bg in enumerate(is_bg) if not bg]
                idx_bg = [i for i, bg in enumerate(is_bg) if bg]

                if t_masked:
                    line, = self.ax.plot(t_masked, r_masked, marker='o', linestyle='-', label=f"BSRH B{b} (Masked)", picker=5)
                    # store metadata
                    line.meta = {'bunch': b, 'indices': idx_masked, 'type': 'masked'}
                if t_bg:
                    line, = self.ax.plot(t_bg, r_bg, marker='x', linestyle='', label=f"BSRH B{b} (BG)", picker=5)
                    line.meta = {'bunch': b, 'indices': idx_bg, 'type': 'bg'}

        # Plot BWS
        bws_data = self.results[plane]['bws']
        for b in bunches:
            if b in bws_data:
                times = bws_data[b]['time']
                rpis = bws_data[b]['rpi']
                if times:
                    self.ax.plot(times, rpis, marker='s', linestyle='--', label=f"BWS B{b}")

        if bunches:
            self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def on_pick(self, event):
        if not hasattr(event.artist, 'meta'):
            return

        meta = event.artist.meta
        ind = event.ind[0] # first clicked point index in the drawn line
        original_idx = meta['indices'][ind]
        bunch = meta['bunch']

        plane = "H" if self.plane_combo.currentText() == "Horizontal" else "V"
        data = self.results[plane]['bsrh'][bunch]

        filepath = data['filepath'][original_idx]
        event_idx = data['event_idx'][original_idx]
        mX = data['maskX'][original_idx]
        mY = data['maskY'][original_idx]
        mR = data['maskR'][original_idx]

        # We will implement PointDetailsDialog in the next step
        # This will open the dialog.
        dialog = PointDetailsDialog(filepath, event_idx, plane, mX, mY, mR, self)
        dialog.exec_()

if __name__ == '__main__':
    try:
        import pyjapc
        japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
    except Exception as e:
        print(f"Warning: Could not initialize pyjapc. Data loading might fail. ({e})")

    app = QApplication(sys.argv)
    gui = AnalysisOfflineGUI()
    gui.show()
    sys.exit(app.exec_())
