import os
import sys
import threading
import time
import math
import zstandard as zstd
import dill as pickle
import numpy as np
import datetime as dt
import copy
from scipy.optimize import curve_fit

import random
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QProgressBar, QMessageBox, QDialog, QSlider, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Utility functions for fitting
def gaussian(x, amp, cen, wid, const):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2)) + const

def fit_gaussian(x, y, p0, bounds=(-np.inf, np.inf)):
    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=p0, bounds=bounds, maxfev=2000)
        return popt
    except Exception:
        return None

# Worker thread for analyzing data
class AnalysisThread(QThread):
    # Signals for updating GUI
    progress_signal = pyqtSignal(int, int) # Current file, total files
    update_plot_signal = pyqtSignal(dict, int) # Emits the grouped data and total BSRH files
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, bsrh_path, bws_h_path, bws_v_path, motor_name, test_mode=False):
        super().__init__()
        self.bsrh_path = bsrh_path
        self.bws_h_path = bws_h_path
        self.bws_v_path = bws_v_path
        self.motor_name = motor_name
        self.test_mode = test_mode
        self.running = True

        # Keep track of processed files
        self.processed_bws_h = set()
        self.processed_bws_v = set()
        self.processed_bsrh = set()

        # Keep track of continuous data
        self.bws_data_h = []
        self.bws_data_v = []
        self.bsrh_grouped_data = {}
        self.total_bsrh_files = 0
        self.previous_motor_pos = None

        self.start_limit = 0
        self.end_limit = float('inf')

        # Buffer to wait 2 seconds to ensure motor isn't moving
        self.event_buffer = []

        self.dctx = zstd.ZstdDecompressor()

    def update_bounds(self, start_idx, end_idx):
        self.start_limit = start_idx
        self.end_limit = end_idx

    def stop(self):
        self.running = False

    def run(self):
        self.status_signal.emit("Started real-time analysis...")
        while self.running:
            bws_h_updated = False
            bws_v_updated = False
            if self.bws_h_path:
                bws_h_updated = self.process_new_bws(self.bws_h_path, self.processed_bws_h, self.bws_data_h, "Horizontal")
            if self.bws_v_path:
                bws_v_updated = self.process_new_bws(self.bws_v_path, self.processed_bws_v, self.bws_data_v, "Vertical")

            bsrh_updated = self.process_new_bsrh(self.bsrh_path)

            if bws_h_updated or bws_v_updated or bsrh_updated:
                self.update_plot_signal.emit(copy.deepcopy(self.bsrh_grouped_data), self.total_bsrh_files)

            time.sleep(1.0) # Poll every 1 second

        self.status_signal.emit("Analysis stopped.")
        self.finished_signal.emit()

    def process_new_bws(self, directory, processed_set, data_list, plane):
        """ Processes newly added BWS files and extracts beam sizes for each bunch and timestamp """
        updated = False
        if not directory or not os.path.isdir(directory):
            return updated

        all_files = sorted([f for f in os.listdir(directory) if f.endswith('.pkl.zst')])
        if len(all_files) == 0:
            return updated

        for i, filename in enumerate(all_files):
            if not self.running: break

            # Note: We don't apply UI slider limits to BWS because they are matched by timestamp to BSRH.
            # However, we still track absolute file parsing.
            if filename in processed_set:
                continue

            self.status_signal.emit(f"Processing new BWS {plane} file: {filename}")

            filepath = os.path.join(directory, filename)

            data = None
            for attempt in range(5):
                try:
                    with open(filepath, 'rb') as f:
                        with self.dctx.stream_reader(f) as reader:
                            data = pickle.load(reader)
                    break # Success
                except Exception as e:
                    if attempt < 4:
                        time.sleep(0.5) # Wait for file to be fully written
                    else:
                        print(f"Error reading BWS file {filename} after 5 attempts: {e}")

            if data is None:
                continue # Skip processing this file and try again next loop

            try:
                timestamp = dt.datetime.fromtimestamp(data.value['gmtTimestampAcquireIn_ns'] / 1e9)

                proj_in = data.value['projDataSet2']
                pos_in = data.value['projPositionSet2'] / 1000.0
                pts_in = int(data.value['nbPtsPerProjInSet2'])

                proj_out = data.value['projDataSet1']
                pos_out = data.value['projPositionSet1'] / 1000.0
                pts_out = int(data.value['nbPtsPerProjInSet1'])

                bunch_list = data.value['bunchList']

                bunches_sizes = {}

                # Iterate over bunches based on the bunchList
                max_bunch = int(len(proj_in) / pts_in)
                for k in range(min(max_bunch, len(bunch_list))):
                    bunch_num = bunch_list[k]

                    # Enforce realistic bounds to prevent variance explosion on noisy fits
                    # Amplitude: [0, inf], Center: [-30, 30], Width: [0.01, 15], Const: [-inf, inf]
                    bws_bounds = ([0, -30.0, 0.01, -np.inf], [np.inf, 30.0, 15.0, np.inf])

                    # IN scan
                    y_in = proj_in[k * pts_in:(k + 1) * pts_in]
                    x_in = pos_in[:pts_in]
                    # Simple moving average to smooth
                    y_in = y_in - np.mean(y_in[:10])
                    popt_in = fit_gaussian(x_in, y_in, [max(y_in), np.mean(x_in), 0.28, np.min(y_in)], bounds=bws_bounds)

                    # OUT scan
                    y_out = proj_out[k * pts_out:(k + 1) * pts_out]
                    x_out = pos_out[:pts_out]
                    y_out = y_out - np.mean(y_out[:10])
                    popt_out = fit_gaussian(x_out, y_out, [max(y_out), np.mean(x_out), 0.28, np.min(y_out)], bounds=bws_bounds)

                    size_in = abs(popt_in[2]) if popt_in is not None else None
                    size_out = abs(popt_out[2]) if popt_out is not None else None

                    if size_in is not None and size_out is not None:
                        bunches_sizes[bunch_num] = (size_in + size_out) / 2.0
                    elif size_in is not None:
                        bunches_sizes[bunch_num] = size_in
                    elif size_out is not None:
                        bunches_sizes[bunch_num] = size_out

                data_list.append({
                    'timestamp': timestamp,
                    'bunches': bunches_sizes
                })
                processed_set.add(filename)
                updated = True
            except Exception as e:
                print(f"Error processing BWS file {filename}: {e}")

        # Sort BWS measurements by timestamp
        data_list.sort(key=lambda x: x['timestamp'])
        return updated

    def process_new_bsrh(self, directory):
        """ Processes BSRH files, extracts beam sizes, and groups them by motor position """
        updated = False

        if not os.path.isdir(directory):
            self.status_signal.emit(f"BSRH directory not found: {directory}")
            return updated

        # Get all files to establish absolute chronological indexing
        all_files = sorted([f for f in os.listdir(directory) if f.endswith('.pkl.zst')])
        self.total_bsrh_files = len(all_files)

        if self.total_bsrh_files == 0:
            return updated

        # Determine the list of files to actively process this sweep
        files_to_process = []
        for i, filename in enumerate(all_files):
            if i < self.start_limit or i > self.end_limit:
                continue
            if filename not in self.processed_bsrh:
                files_to_process.append((i, filename))

        sweep_total = len(files_to_process)
        if sweep_total == 0:
            return updated

        for sweep_idx, (current_file_idx, filename) in enumerate(files_to_process):
            if not self.running: break

            self.progress_signal.emit(sweep_idx + 1, sweep_total)
            self.status_signal.emit(f"Processing new BSRH file: {filename} ({sweep_idx + 1}/{sweep_total})")

            filepath = os.path.join(directory, filename)

            bsrh_events = None
            for attempt in range(5):
                try:
                    with open(filepath, 'rb') as f:
                        with self.dctx.stream_reader(f) as reader:
                            bsrh_events = pickle.load(reader)
                    break # Success
                except Exception as e:
                    if attempt < 4:
                        time.sleep(0.5) # Wait for file to be fully written
                    else:
                        print(f"Error reading BSRH file {filename} after 5 attempts: {e}")

            if bsrh_events is None:
                continue # Skip processing this file and try again next loop

            try:
                for event in bsrh_events:
                    timestamp = event.timestamp.replace(tzinfo=None)
                    motor_pos = None
                    if self.motor_name in event.device_data:
                        motor_data = event.device_data[self.motor_name]
                        if isinstance(motor_data, dict) and 'position' in motor_data:
                            motor_pos = motor_data['position']
                        else:
                            motor_pos = motor_data

                    if motor_pos is None:
                        continue # Skip if motor position not found

                    # Filter out transient motor positions (rubbish data while moving)
                    if self.previous_motor_pos is None or motor_pos != self.previous_motor_pos:
                        self.previous_motor_pos = motor_pos
                        continue # Discard this event as it represents movement

                    bunch_slot = event.device_data.get('GATE-INT', {}).get('delaySlots', None)
                    if bunch_slot is None:
                        continue

                    # Process image to find BSRH beam sizes
                    image = event.cam_data['image2D']

                    # Horizontal projection
                    proj_x = np.sum(image, axis=0)
                    proj_x = proj_x - np.mean(proj_x[:50])
                    x_temp = np.arange(len(proj_x))
                    popt_h = fit_gaussian(x_temp, proj_x, [np.max(proj_x), np.argmax(proj_x), 15.0, np.min(proj_x)])
                    bsrh_size_h = abs(popt_h[2]) if popt_h is not None else None

                    # Vertical projection
                    proj_y = np.sum(image, axis=1)
                    proj_y = proj_y - np.mean(proj_y[:50])
                    y_temp = np.arange(len(proj_y))
                    popt_v = fit_gaussian(y_temp, proj_y, [np.max(proj_y), np.argmax(proj_y), 15.0, np.min(proj_y)])
                    bsrh_size_v = abs(popt_v[2]) if popt_v is not None else None

                    if bsrh_size_h is None and bsrh_size_v is None:
                        continue

                    # Find closest BWS measurements
                    closest_bws_h = self.find_closest_bws(self.bws_data_h, timestamp, bunch_slot) if self.bws_h_path else None
                    closest_bws_v = self.find_closest_bws(self.bws_data_v, timestamp, bunch_slot) if self.bws_v_path else None

                    if motor_pos not in self.bsrh_grouped_data:
                        self.bsrh_grouped_data[motor_pos] = {
                            'bsrh_sizes_h': [], 'bws_sizes_h': [],
                            'bsrh_sizes_v': [], 'bws_sizes_v': [],
                            'bunches': [], 'file_indices': []
                        }

                    self.bsrh_grouped_data[motor_pos]['bsrh_sizes_h'].append(bsrh_size_h)
                    self.bsrh_grouped_data[motor_pos]['bws_sizes_h'].append(closest_bws_h)

                    self.bsrh_grouped_data[motor_pos]['bsrh_sizes_v'].append(bsrh_size_v)
                    self.bsrh_grouped_data[motor_pos]['bws_sizes_v'].append(closest_bws_v)

                    self.bsrh_grouped_data[motor_pos]['bunches'].append(bunch_slot)
                    self.bsrh_grouped_data[motor_pos]['file_indices'].append(current_file_idx)

                self.processed_bsrh.add(filename)
                updated = True

            except Exception as e:
                print(f"Error processing BSRH file {filename}: {e}")

        return updated

    def find_closest_bws(self, bws_data_list, bsrh_time, bunch_slot):
        """ Finds the closest BWS measurement for a given timestamp and bunch slot """
        if not bws_data_list:
            return None

        if self.test_mode:
            # In test mode, completely ignore timestamps and bunches, and just pick a random available size
            random_scan = random.choice(bws_data_list)
            if random_scan['bunches']:
                # Pick a random bunch's size from this scan
                random_bunch = random.choice(list(random_scan['bunches'].keys()))
                return random_scan['bunches'][random_bunch]
            return None

        best_diff = float('inf')
        best_size = None

        for bws in bws_data_list:
            if bunch_slot in bws['bunches']:
                diff = abs((bsrh_time - bws['timestamp']).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_size = bws['bunches'][bunch_slot]

        return best_size

class SecondaryPlotDialog(QDialog):
    def __init__(self, motor_pos, data, is_horizontal, is_vertical, parent=None):
        super().__init__(parent)
        plane_str = "Horizontal" if is_horizontal else "Vertical"
        self.setWindowTitle(f"Correlation Analysis ({plane_str}) for Motor Position: {motor_pos}")
        self.resize(800, 600)

        self.layout = QVBoxLayout(self)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)

        self.motor_pos = motor_pos
        self.data = data
        self.is_horizontal = is_horizontal
        self.is_vertical = is_vertical
        self.parent_gui = parent

        # Connect pick event to click points
        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.plot_data()

    def plot_data(self):
        self.ax.clear()

        self.valid_data_indices = []

        if self.is_horizontal:
            # Filter out None values from both BSRH and BWS
            self.valid_data_indices = [i for i, bws in enumerate(self.data['bws_sizes_h']) if bws is not None and self.data['bsrh_sizes_h'][i] is not None]
            # Apply Beta-Function scaling BSRH/BWS coefficient
            bws_sizes = [self.data['bws_sizes_h'][i] * 1.023 for i in self.valid_data_indices]
            bsrh_sizes = [self.data['bsrh_sizes_h'][i] for i in self.valid_data_indices]
        elif self.is_vertical:
            self.valid_data_indices = [i for i, bws in enumerate(self.data['bws_sizes_v']) if bws is not None and self.data['bsrh_sizes_v'][i] is not None]
            # Apply Beta-Function scaling BSRH/BWS coefficient
            bws_sizes = [self.data['bws_sizes_v'][i] * 0.936 for i in self.valid_data_indices]
            bsrh_sizes = [self.data['bsrh_sizes_v'][i] for i in self.valid_data_indices]
        else:
            return

        if len(bws_sizes) == 0:
            self.ax.text(0.5, 0.5, 'No valid BWS data to correlate.', horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        # We need to plot BSRH_size^2 vs BWS_size^2
        x_data = np.array(bws_sizes)**2
        y_data = np.array(bsrh_sizes)**2

        color = 'blue' if self.is_horizontal else 'red'
        plane_str = "Horizontal" if self.is_horizontal else "Vertical"

        self.scatter_plot = self.ax.scatter(x_data, y_data, color=color, label=f'{plane_str} Data Points', picker=10)

        # Perform linear fit
        try:
            # y = mx + c
            coeffs = np.polyfit(x_data, y_data, 1)
            slope, intercept = coeffs[0], coeffs[1]

            fit_line = slope * x_data + intercept
            self.ax.plot(x_data, fit_line, color='black', linestyle='--', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')

            # Calculate Magnification and PSF
            if slope > 0:
                pixel_size = np.sqrt(1 / slope)
                magnification = 1 / pixel_size
                if intercept > 0:
                    psf = np.sqrt(intercept / slope)
                else:
                    psf = 0.0 # Or nan if negative intercept

                info_text = (f"Slope: {slope:.3f}\n"
                             f"Intercept: {intercept:.3f}\n"
                             f"Pixel Size: {pixel_size*1000:.2f} μm/px\n"
                             f"Magnification: {magnification:.2f}\n"
                             f"PSF: {psf*1000:.2f} μm")

                self.ax.text(0.05, 0.95, info_text, transform=self.ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"Fit error: {e}")

        self.ax.set_xlabel('Scaled BWS Size Squared (Expected BSRH Size)')
        self.ax.set_ylabel('BSRH Size Squared (px^2)')
        self.ax.set_title(f'{plane_str} BSRH vs Scaled BWS Beam Size Squared')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def on_pick(self, event):
        try:
            if event.artist != self.scatter_plot:
                return

            ind = event.ind
            if len(ind) > 0:
                index = ind[0]

                # The 'index' corresponds to the element in 'self.valid_data_indices'
                actual_data_idx = self.valid_data_indices[index]

                file_idx = self.data['file_indices'][actual_data_idx]
                bunch_slot = self.data['bunches'][actual_data_idx]

                # Add to parent's ignored points
                if self.parent_gui:
                    self.parent_gui.ignored_points.add((self.motor_pos, file_idx, bunch_slot))

                    # We also need to remove it from the local data so it disappears immediately on re-plot
                    # Just nullify it so valid_idx filter catches it
                    if self.is_horizontal:
                        self.data['bsrh_sizes_h'][actual_data_idx] = None
                    if self.is_vertical:
                        self.data['bsrh_sizes_v'][actual_data_idx] = None

                    # Re-plot the dialog
                    self.plot_data()

                    # Tell parent to redraw the main plot
                    if self.parent_gui.grouped_data:
                        self.parent_gui.redraw_plot()
        except Exception as e:
            print(f"Error ignoring point on pick event: {e}")

class AnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSRH/BWS Analysis GUI")
        self.resize(1000, 700)

        self.layout = QVBoxLayout(self)

        # Input Layout
        input_layout = QVBoxLayout()

        bsrh_layout = QHBoxLayout()
        self.bsrh_btn = QPushButton("Select BSRH Dir")
        self.bsrh_btn.clicked.connect(self.select_bsrh_dir)
        self.bsrh_lbl = QLabel("No directory selected")
        bsrh_layout.addWidget(self.bsrh_btn)
        bsrh_layout.addWidget(self.bsrh_lbl)

        bws_h_layout = QHBoxLayout()
        self.bws_h_btn = QPushButton("Select BWS Horizontal Dir (Optional)")
        self.bws_h_btn.clicked.connect(self.select_bws_h_dir)
        self.bws_h_lbl = QLabel("No directory selected")
        bws_h_layout.addWidget(self.bws_h_btn)
        bws_h_layout.addWidget(self.bws_h_lbl)

        bws_v_layout = QHBoxLayout()
        self.bws_v_btn = QPushButton("Select BWS Vertical Dir (Optional)")
        self.bws_v_btn.clicked.connect(self.select_bws_v_dir)
        self.bws_v_lbl = QLabel("No directory selected")
        bws_v_layout.addWidget(self.bws_v_btn)
        bws_v_layout.addWidget(self.bws_v_lbl)

        motor_layout = QHBoxLayout()
        self.motor_input = QLineEdit()
        self.motor_input.setPlaceholderText("Enter motor name (e.g., M3-X)")
        motor_layout.addWidget(QLabel("Motor:"))
        motor_layout.addWidget(self.motor_input)

        self.test_mode_checkbox = QCheckBox("Test Mode (Randomize BWS Matching)")
        self.test_mode_checkbox.setToolTip("If checked, ignores timestamps/bunches and randomly pairs available BWS data with BSRH events to test GUI components.")
        motor_layout.addWidget(self.test_mode_checkbox)

        input_layout.addLayout(bsrh_layout)
        input_layout.addLayout(bws_h_layout)
        input_layout.addLayout(bws_v_layout)
        input_layout.addLayout(motor_layout)

        self.layout.addLayout(input_layout)

        # Control Layout
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.clicked.connect(self.start_analysis)
        self.stop_btn = QPushButton("Stop Analysis")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.status_lbl = QLabel("Ready")

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.status_lbl)

        self.layout.addLayout(control_layout)

        # Slider Layout
        slider_layout = QVBoxLayout()

        start_slider_row = QHBoxLayout()
        self.start_lbl = QLabel("Start File: 0")
        self.start_lbl.setMinimumWidth(100)
        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.setMinimum(0)
        self.start_slider.setMaximum(0)
        self.start_slider.setEnabled(False)
        self.start_slider.valueChanged.connect(self.on_sliders_changed)
        start_slider_row.addWidget(self.start_lbl)
        start_slider_row.addWidget(self.start_slider)

        end_slider_row = QHBoxLayout()
        self.end_lbl = QLabel("End File: Last")
        self.end_lbl.setMinimumWidth(100)
        self.end_slider = QSlider(Qt.Horizontal)
        self.end_slider.setMinimum(0)
        self.end_slider.setMaximum(0)
        self.end_slider.setEnabled(False)
        self.end_slider.valueChanged.connect(self.on_sliders_changed)
        end_slider_row.addWidget(self.end_lbl)
        end_slider_row.addWidget(self.end_slider)

        slider_layout.addLayout(start_slider_row)
        slider_layout.addLayout(end_slider_row)
        self.layout.addLayout(slider_layout)

        # Main Plot Canvas
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax_beam = self.figure.add_subplot(211)
        self.ax_beam.set_xlabel("Motor Position")
        self.ax_beam.set_ylabel("Average BSRH Beam Size (px)")
        self.ax_beam.set_title("Average BSRH Beam Size vs Motor Position")

        self.ax_params = self.figure.add_subplot(212)
        self.ax_params_twin = self.ax_params.twinx()
        self.ax_params.set_xlabel("Motor Position")
        self.ax_params.set_ylabel("PSF (\u03bcm)")
        self.ax_params_twin.set_ylabel("Pixel Size (\u03bcm/px)")

        self.figure.tight_layout()
        self.canvas.draw()

        # Connect pick event for scatter plot
        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.bsrh_path = ""
        self.bws_h_path = ""
        self.bws_v_path = ""
        self.worker = None
        self.grouped_data = {}
        self.total_files = 0
        self.scatter_plot_h = None
        self.scatter_plot_v = None
        self.active_dialogs = []
        self.ignored_points = set() # Store tuples of (motor_pos, file_idx, bunch_slot)

    def select_bsrh_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select BSRH Directory")
        if dir_path:
            self.bsrh_path = dir_path
            self.bsrh_lbl.setText(dir_path)

            # Immediately scan directory to configure sliders
            files = sorted([f for f in os.listdir(dir_path) if f.endswith('.pkl.zst')])
            self.total_files = len(files)
            if self.total_files > 0:
                max_idx = self.total_files - 1
                self.start_slider.setMaximum(max_idx)
                self.start_slider.setValue(0)
                self.start_slider.setEnabled(True)

                self.end_slider.setMaximum(max_idx)
                self.end_slider.setValue(max_idx)
                self.end_slider.setEnabled(True)

                self.start_lbl.setText("Start File: 0")
                self.end_lbl.setText("End File: Last")

    def select_bws_h_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select BWS Horizontal Directory")
        if dir_path:
            self.bws_h_path = dir_path
            self.bws_h_lbl.setText(dir_path)

    def select_bws_v_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select BWS Vertical Directory")
        if dir_path:
            self.bws_v_path = dir_path
            self.bws_v_lbl.setText(dir_path)

    def start_analysis(self):
        motor_name = self.motor_input.text().strip()
        if not self.bsrh_path or not motor_name:
            QMessageBox.warning(self, "Input Error", "Please select at least the BSRH directory and provide a motor name.")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.motor_input.setEnabled(False)

        self.grouped_data.clear()

        # We don't clear total_files or sliders here because they are initialized during directory selection

        self.ax_beam.clear()
        self.ax_beam.set_xlabel(f"Motor Position ({motor_name})")
        self.ax_beam.set_ylabel("Average BSRH Beam Size (px)")
        self.ax_beam.set_title("Average BSRH Beam Size vs Motor Position")

        self.ax_params.clear()
        self.ax_params_twin.clear()
        self.ax_params.set_xlabel(f"Motor Position ({motor_name})")
        self.ax_params.set_ylabel("PSF (\u03bcm)")
        self.ax_params_twin.set_ylabel("Pixel Size (\u03bcm/px)")

        self.canvas.draw()

        test_mode_enabled = self.test_mode_checkbox.isChecked()
        self.worker = AnalysisThread(self.bsrh_path, self.bws_h_path, self.bws_v_path, motor_name, test_mode=test_mode_enabled)

        # Pass the initial slider constraints down to the worker thread
        start_idx = self.start_slider.value()
        end_idx = self.end_slider.value()
        if end_idx == self.end_slider.maximum():
            end_idx = float('inf')
        self.worker.update_bounds(start_idx, end_idx)

        self.worker.progress_signal.connect(self.update_progress)
        self.worker.status_signal.connect(self.update_status)
        self.worker.update_plot_signal.connect(self.update_plot)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.start()

    def stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_lbl.setText("Stopping...")

    def update_progress(self, current, total):
        if total > 0:
            val = int((current / total) * 100)
            self.progress_bar.setValue(val)

    def update_status(self, text):
        self.status_lbl.setText(text)

    def on_sliders_changed(self):
        # Update labels
        start_val = self.start_slider.value()
        self.start_lbl.setText(f"Start File: {start_val}")

        end_val = self.end_slider.value()
        if end_val >= self.total_files - 1 and self.total_files > 0:
            self.end_lbl.setText("End File: Last")
            worker_end_val = float('inf')
        else:
            self.end_lbl.setText(f"End File: {end_val}")
            worker_end_val = end_val

        # Send dynamic limits to worker thread if it exists
        if self.worker and self.worker.isRunning():
            self.worker.update_bounds(start_val, worker_end_val)

        # Re-trigger plot with current data
        if self.grouped_data:
            self.redraw_plot()

    def update_plot(self, grouped_data, total_bsrh_files):
        self.grouped_data = grouped_data
        self.total_files = total_bsrh_files

        # Update sliders
        was_start_max = (self.start_slider.value() == self.start_slider.maximum())
        was_end_max = (self.end_slider.value() == self.end_slider.maximum())

        if total_bsrh_files > 0:
            max_idx = total_bsrh_files - 1

            # Temporarily block signals to avoid multiple redraws
            self.start_slider.blockSignals(True)
            self.end_slider.blockSignals(True)

            self.start_slider.setMaximum(max_idx)
            self.end_slider.setMaximum(max_idx)

            # If end slider was at the maximum before, keep it at the new maximum
            if was_end_max:
                self.end_slider.setValue(max_idx)
                self.end_lbl.setText("End File: Last")

                # Push the "inf" bound down to the worker thread so it keeps consuming seamlessly
                if self.worker and self.worker.isRunning():
                    self.worker.update_bounds(self.start_slider.value(), float('inf'))

            self.start_slider.blockSignals(False)
            self.end_slider.blockSignals(False)

        self.redraw_plot()

    def redraw_plot(self):
        motor_positions_h = []
        avg_sizes_h = []
        psf_h_list = []
        pix_h_list = []

        motor_positions_v = []
        avg_sizes_v = []
        psf_v_list = []
        pix_v_list = []

        start_idx = self.start_slider.value()
        end_idx = self.end_slider.value()

        # If end_slider is at max, it signifies "Last", so include all remaining
        if end_idx == self.end_slider.maximum():
            end_idx = float('inf')

        for pos, data in self.grouped_data.items():
            valid_h = []
            valid_bws_h = []
            bunches_h = {}

            valid_v = []
            valid_bws_v = []
            bunches_v = {}

            for i, f_idx in enumerate(data['file_indices']):
                if start_idx <= f_idx <= end_idx:
                    bunch_slot = data['bunches'][i]

                    # Ensure we skip ignored points
                    if (pos, f_idx, bunch_slot) in self.ignored_points:
                        continue

                    if i < len(data['bsrh_sizes_h']) and data['bsrh_sizes_h'][i] is not None:
                        val_h = data['bsrh_sizes_h'][i]
                        valid_h.append(val_h)
                        if bunch_slot not in bunches_h: bunches_h[bunch_slot] = []
                        bunches_h[bunch_slot].append(val_h)

                        if i < len(data['bws_sizes_h']) and data['bws_sizes_h'][i] is not None:
                            valid_bws_h.append(data['bws_sizes_h'][i])

                    if i < len(data['bsrh_sizes_v']) and data['bsrh_sizes_v'][i] is not None:
                        val_v = data['bsrh_sizes_v'][i]
                        valid_v.append(val_v)
                        if bunch_slot not in bunches_v: bunches_v[bunch_slot] = []
                        bunches_v[bunch_slot].append(val_v)

                        if i < len(data['bws_sizes_v']) and data['bws_sizes_v'][i] is not None:
                            valid_bws_v.append(data['bws_sizes_v'][i])

            if len(valid_h) >= 4:
                motor_positions_h.append(pos)

                # Calculate the grand average of per-bunch averages
                bunch_averages_h = [np.mean(vals) for vals in bunches_h.values()]
                avg_sizes_h.append(np.mean(bunch_averages_h))

                # Compute PSF and Pixel Size for this position
                if valid_bws_h and len(valid_bws_h) == len(valid_h):
                    x_data = np.array([bws * 1.023 for bws in valid_bws_h])**2
                    y_data = np.array(valid_h)**2
                    try:
                        coeffs = np.polyfit(x_data, y_data, 1)
                        slope, intercept = coeffs[0], coeffs[1]
                        pix = np.sqrt(1 / slope) if slope > 0 else np.nan
                        psf = np.sqrt(intercept / slope) if slope > 0 and intercept > 0 else 0.0
                        psf_h_list.append(psf * 1000) # to um
                        pix_h_list.append(pix * 1000) # to um/px
                    except:
                        psf_h_list.append(np.nan)
                        pix_h_list.append(np.nan)
                else:
                    psf_h_list.append(np.nan)
                    pix_h_list.append(np.nan)

            if len(valid_v) >= 4:
                motor_positions_v.append(pos)

                # Calculate the grand average of per-bunch averages
                bunch_averages_v = [np.mean(vals) for vals in bunches_v.values()]
                avg_sizes_v.append(np.mean(bunch_averages_v))

                if valid_bws_v and len(valid_bws_v) == len(valid_v):
                    x_data = np.array([bws * 0.936 for bws in valid_bws_v])**2
                    y_data = np.array(valid_v)**2
                    try:
                        coeffs = np.polyfit(x_data, y_data, 1)
                        slope, intercept = coeffs[0], coeffs[1]
                        pix = np.sqrt(1 / slope) if slope > 0 else np.nan
                        psf = np.sqrt(intercept / slope) if slope > 0 and intercept > 0 else 0.0
                        psf_v_list.append(psf * 1000) # to um
                        pix_v_list.append(pix * 1000) # to um/px
                    except:
                        psf_v_list.append(np.nan)
                        pix_v_list.append(np.nan)
                else:
                    psf_v_list.append(np.nan)
                    pix_v_list.append(np.nan)

        self.ax_beam.clear()
        self.ax_beam.set_xlabel(f"Motor Position ({self.motor_input.text().strip()})")
        self.ax_beam.set_ylabel("Average BSRH Beam Size (px)")
        self.ax_beam.set_title("Average BSRH Beam Size vs Motor Position")

        self.ax_params.clear()
        self.ax_params_twin.clear()

        # Clearing twinx resets its ticks and labels to the left, we must explicitly reset them to the right
        self.ax_params_twin.yaxis.tick_right()
        self.ax_params_twin.yaxis.set_label_position("right")

        self.ax_params.set_xlabel(f"Motor Position ({self.motor_input.text().strip()})")
        self.ax_params.set_ylabel("PSF (\u03bcm)")
        self.ax_params_twin.set_ylabel("Pixel Size (\u03bcm/px)")

        # Determine if points should be clickable (if we have corresponding BWS data)
        picker_h = 5 if self.bws_h_path else None
        picker_v = 5 if self.bws_v_path else None

        if motor_positions_h:
            self.scatter_plot_h = self.ax_beam.scatter(motor_positions_h, avg_sizes_h, c='blue', label='Horizontal', picker=picker_h)
            # Plot PSF/Pix
            self.ax_params.plot(motor_positions_h, psf_h_list, 'bo-', label='PSF H')
            self.ax_params_twin.plot(motor_positions_h, pix_h_list, 'b^--', label='Pix H')

        if motor_positions_v:
            self.scatter_plot_v = self.ax_beam.scatter(motor_positions_v, avg_sizes_v, c='red', label='Vertical', picker=picker_v)
            # Plot PSF/Pix
            self.ax_params.plot(motor_positions_v, psf_v_list, 'ro-', label='PSF V')
            self.ax_params_twin.plot(motor_positions_v, pix_v_list, 'r^--', label='Pix V')

        if motor_positions_h or motor_positions_v:
            self.ax_beam.legend()
            self.ax_beam.grid(True)

            # Combine legends for ax_params and twin
            lines_1, labels_1 = self.ax_params.get_legend_handles_labels()
            lines_2, labels_2 = self.ax_params_twin.get_legend_handles_labels()
            if lines_1 or lines_2:
                self.ax_params.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
            self.ax_params.grid(True)

        self.canvas.draw()

    def analysis_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.motor_input.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_lbl.setText("Analysis finished.")

    def on_pick(self, event):
        is_horizontal = False
        is_vertical = False

        if event.artist == self.scatter_plot_h:
            is_horizontal = True
        elif event.artist == self.scatter_plot_v:
            is_vertical = True
        else:
            return

        ind = event.ind
        if len(ind) > 0:
            index = ind[0]
            # Retrieve motor position from the plot data
            offsets = event.artist.get_offsets()
            motor_pos = offsets[index][0]

            # Find the closest matching motor_pos in our grouped_data keys
            best_pos = None
            min_diff = float('inf')
            for pos in self.grouped_data.keys():
                diff = abs(pos - motor_pos)
                if diff < min_diff:
                    min_diff = diff
                    best_pos = pos

            if best_pos is not None:
                data = self.grouped_data[best_pos]

                # Filter data for the dialog based on sliders
                start_idx = self.start_slider.value()
                end_idx = self.end_slider.value()
                if end_idx == self.end_slider.maximum():
                    end_idx = float('inf')

                filtered_data = {
                    'bsrh_sizes_h': [], 'bws_sizes_h': [],
                    'bsrh_sizes_v': [], 'bws_sizes_v': [],
                    'bunches': [], 'file_indices': []
                }

                for i, f_idx in enumerate(data['file_indices']):
                    if start_idx <= f_idx <= end_idx:
                        bunch_slot = data['bunches'][i]
                        if (best_pos, f_idx, bunch_slot) in self.ignored_points:
                            continue

                        if i < len(data['bsrh_sizes_h']):
                            filtered_data['bsrh_sizes_h'].append(data['bsrh_sizes_h'][i])
                        if i < len(data['bws_sizes_h']):
                            filtered_data['bws_sizes_h'].append(data['bws_sizes_h'][i])
                        if i < len(data['bsrh_sizes_v']):
                            filtered_data['bsrh_sizes_v'].append(data['bsrh_sizes_v'][i])
                        if i < len(data['bws_sizes_v']):
                            filtered_data['bws_sizes_v'].append(data['bws_sizes_v'][i])
                        filtered_data['bunches'].append(bunch_slot)
                        filtered_data['file_indices'].append(f_idx)

                dialog = SecondaryPlotDialog(best_pos, filtered_data, is_horizontal, is_vertical, self)
                self.active_dialogs.append(dialog)
                dialog.show()

                # Clean up any closed dialogs to prevent memory leaks
                self.active_dialogs = [d for d in self.active_dialogs if d.isVisible()]

if __name__ == "__main__":
    # Required to initialize the Java VM so that pickle can load JAPC objects
    try:
        import pyjapc
        japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
    except Exception as e:
        print(f"Warning: Could not initialize pyjapc. Data loading might fail. ({e})")

    app = QApplication(sys.argv)
    gui = AnalysisGUI()
    gui.show()
    sys.exit(app.exec_())
