import os
import sys
import threading
import time
import math
import zstandard as zstd
import dill as pickle
import numpy as np
import datetime as dt
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QProgressBar, QMessageBox, QDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Utility functions for fitting
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def fit_gaussian(x, y, p0):
    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=p0, maxfev=2000)
        return popt
    except Exception:
        return None

# Worker thread for analyzing data
class AnalysisThread(QThread):
    # Signals for updating GUI
    progress_signal = pyqtSignal(int, int) # Current file, total files
    update_plot_signal = pyqtSignal(dict) # Emits the grouped data
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, bsrh_path, bws_path, motor_name):
        super().__init__()
        self.bsrh_path = bsrh_path
        self.bws_path = bws_path
        self.motor_name = motor_name
        self.running = True

        # Keep track of processed files
        self.processed_bws = set()
        self.processed_bsrh = set()

        # Keep track of continuous data
        self.bws_data = []
        self.bsrh_grouped_data = {}

        self.dctx = zstd.ZstdDecompressor()

    def stop(self):
        self.running = False

    def run(self):
        self.status_signal.emit("Started real-time analysis...")
        while self.running:
            bws_updated = self.process_new_bws(self.bws_path)
            bsrh_updated = self.process_new_bsrh(self.bsrh_path)

            if bws_updated or bsrh_updated:
                # Re-calculate matches with new BWS data or simply emit new BSRH groups
                self.update_plot_signal.emit(self.bsrh_grouped_data)

            time.sleep(1.0) # Poll every 1 second

        self.status_signal.emit("Analysis stopped.")
        self.finished_signal.emit()

    def process_new_bws(self, directory):
        """ Processes newly added BWS files and extracts beam sizes for each bunch and timestamp """
        updated = False
        if not os.path.isdir(directory):
            self.status_signal.emit(f"BWS directory not found: {directory}")
            return updated

        files = sorted([f for f in os.listdir(directory) if f.endswith('.pkl.zst') and f not in self.processed_bws])
        total_files = len(files)
        if total_files == 0:
            return updated

        for i, filename in enumerate(files):
            if not self.running: break
            self.status_signal.emit(f"Processing new BWS file: {filename}")

            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'rb') as f:
                    with self.dctx.stream_reader(f) as reader:
                        data = pickle.load(reader)

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

                    # IN scan
                    y_in = proj_in[k * pts_in:(k + 1) * pts_in]
                    x_in = pos_in[:pts_in]
                    # Simple moving average to smooth
                    y_in = y_in - np.mean(y_in[:10])
                    popt_in = fit_gaussian(x_in, y_in, [max(y_in), np.mean(x_in), 0.28])

                    # OUT scan
                    y_out = proj_out[k * pts_out:(k + 1) * pts_out]
                    x_out = pos_out[:pts_out]
                    y_out = y_out - np.mean(y_out[:10])
                    popt_out = fit_gaussian(x_out, y_out, [max(y_out), np.mean(x_out), 0.28])

                    size_in = abs(popt_in[2]) if popt_in else None
                    size_out = abs(popt_out[2]) if popt_out else None

                    if size_in and size_out:
                        bunches_sizes[bunch_num] = (size_in + size_out) / 2.0
                    elif size_in:
                        bunches_sizes[bunch_num] = size_in
                    elif size_out:
                        bunches_sizes[bunch_num] = size_out

                self.bws_data.append({
                    'timestamp': timestamp,
                    'bunches': bunches_sizes
                })
                self.processed_bws.add(filename)
                updated = True
            except Exception as e:
                print(f"Error processing BWS file {filename}: {e}")

        # Sort BWS measurements by timestamp
        self.bws_data.sort(key=lambda x: x['timestamp'])
        return updated

    def process_new_bsrh(self, directory):
        """ Processes BSRH files, extracts beam sizes, and groups them by motor position """
        updated = False

        if not os.path.isdir(directory):
            self.status_signal.emit(f"BSRH directory not found: {directory}")
            return updated

        files = sorted([f for f in os.listdir(directory) if f.endswith('.pkl.zst') and f not in self.processed_bsrh])
        total_files = len(files)
        if total_files == 0:
            return updated

        for i, filename in enumerate(files):
            if not self.running: break
            self.status_signal.emit(f"Processing new BSRH file: {filename}")

            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'rb') as f:
                    with self.dctx.stream_reader(f) as reader:
                        bsrh_events = pickle.load(reader)

                for event in bsrh_events:
                    # Extract timestamp, motor position, and delay slots
                    timestamp = event.timestamp.replace(tzinfo=None)

                    # Get motor position from device_data
                    motor_pos = None
                    if self.motor_name in event.device_data:
                        motor_data = event.device_data[self.motor_name]
                        if isinstance(motor_data, dict) and 'position' in motor_data:
                            motor_pos = motor_data['position']
                        else:
                            motor_pos = motor_data

                    if motor_pos is None:
                        continue # Skip if motor position not found

                    bunch_slot = event.device_data.get('GATE-INT', {}).get('delaySlots', None)
                    if bunch_slot is None:
                        continue

                    # Find closest BWS measurement
                    closest_bws = self.find_closest_bws(timestamp, bunch_slot)
                    if not closest_bws:
                        continue

                    # Process image to find BSRH beam size
                    image = event.cam_data['image2D']
                    # Use horizontal projection for beam size (or average of H and V)
                    # For simplicity, let's use horizontal projection
                    proj_x = np.sum(image, axis=0)
                    proj_x = proj_x - np.mean(proj_x[:50])
                    x_temp = np.arange(len(proj_x))

                    popt = fit_gaussian(x_temp, proj_x, [np.max(proj_x), np.argmax(proj_x), 15.0])

                    if popt:
                        bsrh_size = abs(popt[2])

                        if motor_pos not in self.bsrh_grouped_data:
                            self.bsrh_grouped_data[motor_pos] = {'bsrh_sizes': [], 'bws_sizes': [], 'bunches': []}

                        self.bsrh_grouped_data[motor_pos]['bsrh_sizes'].append(bsrh_size)
                        self.bsrh_grouped_data[motor_pos]['bws_sizes'].append(closest_bws)
                        self.bsrh_grouped_data[motor_pos]['bunches'].append(bunch_slot)

                self.processed_bsrh.add(filename)
                updated = True

            except Exception as e:
                print(f"Error processing BSRH file {filename}: {e}")

        return updated

    def find_closest_bws(self, bsrh_time, bunch_slot):
        """ Finds the closest BWS measurement for a given timestamp and bunch slot """
        best_diff = float('inf')
        best_size = None

        for bws in self.bws_data:
            if bunch_slot in bws['bunches']:
                diff = abs((bsrh_time - bws['timestamp']).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_size = bws['bunches'][bunch_slot]

        return best_size

class SecondaryPlotDialog(QDialog):
    def __init__(self, motor_pos, bsrh_sizes, bws_sizes, bunches, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Correlation Analysis for Motor Position: {motor_pos}")
        self.resize(800, 600)

        self.layout = QVBoxLayout(self)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.plot_data(bsrh_sizes, bws_sizes)

    def plot_data(self, bsrh_sizes, bws_sizes):
        ax = self.figure.add_subplot(111)
        ax.clear()

        # We need to plot BSRH_size^2 vs BWS_size^2
        x_data = np.array(bws_sizes)**2
        y_data = np.array(bsrh_sizes)**2

        ax.scatter(x_data, y_data, color='blue', label='Data Points')

        # Perform linear fit
        try:
            # y = mx + c
            coeffs = np.polyfit(x_data, y_data, 1)
            slope, intercept = coeffs[0], coeffs[1]

            fit_line = slope * x_data + intercept
            ax.plot(x_data, fit_line, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')

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

                ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"Fit error: {e}")

        ax.set_xlabel('BWS Size Squared')
        ax.set_ylabel('BSRH Size Squared (px^2)')
        ax.set_title('BSRH vs BWS Beam Size Squared')
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

class AnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSRH/BWS Analysis GUI")
        self.resize(1000, 700)

        self.layout = QVBoxLayout(self)

        # Input Layout
        input_layout = QHBoxLayout()

        self.bsrh_btn = QPushButton("Select BSRH Dir")
        self.bsrh_btn.clicked.connect(self.select_bsrh_dir)
        self.bsrh_lbl = QLabel("No directory selected")

        self.bws_btn = QPushButton("Select BWS Dir")
        self.bws_btn.clicked.connect(self.select_bws_dir)
        self.bws_lbl = QLabel("No directory selected")

        self.motor_input = QLineEdit()
        self.motor_input.setPlaceholderText("Enter motor name (e.g., M3-X)")

        input_layout.addWidget(self.bsrh_btn)
        input_layout.addWidget(self.bsrh_lbl)
        input_layout.addWidget(self.bws_btn)
        input_layout.addWidget(self.bws_lbl)
        input_layout.addWidget(QLabel("Motor:"))
        input_layout.addWidget(self.motor_input)

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

        # Main Plot Canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Motor Position")
        self.ax.set_ylabel("Average BSRH Beam Size (px)")
        self.ax.set_title("Average BSRH Beam Size vs Motor Position")
        self.canvas.draw()

        # Connect pick event for scatter plot
        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.bsrh_path = ""
        self.bws_path = ""
        self.worker = None
        self.grouped_data = {}
        self.scatter_plot = None

    def select_bsrh_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select BSRH Directory")
        if dir_path:
            self.bsrh_path = dir_path
            self.bsrh_lbl.setText(dir_path)

    def select_bws_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select BWS Directory")
        if dir_path:
            self.bws_path = dir_path
            self.bws_lbl.setText(dir_path)

    def start_analysis(self):
        motor_name = self.motor_input.text().strip()
        if not self.bsrh_path or not self.bws_path or not motor_name:
            QMessageBox.warning(self, "Input Error", "Please select directories and provide a motor name.")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.motor_input.setEnabled(False)

        self.grouped_data.clear()
        self.ax.clear()
        self.ax.set_xlabel(f"Motor Position ({motor_name})")
        self.ax.set_ylabel("Average BSRH Beam Size (px)")
        self.ax.set_title("Average BSRH Beam Size vs Motor Position")
        self.canvas.draw()

        self.worker = AnalysisThread(self.bsrh_path, self.bws_path, motor_name)
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

    def update_plot(self, grouped_data):
        self.grouped_data = grouped_data

        motor_positions = []
        avg_sizes = []

        for pos, data in grouped_data.items():
            motor_positions.append(pos)
            avg_sizes.append(np.mean(data['bsrh_sizes']))

        self.ax.clear()
        self.ax.set_xlabel(f"Motor Position ({self.motor_input.text().strip()})")
        self.ax.set_ylabel("Average BSRH Beam Size (px)")
        self.ax.set_title("Average BSRH Beam Size vs Motor Position")

        if motor_positions:
            # Note: picker=5 means a 5 point tolerance for clicks
            self.scatter_plot = self.ax.scatter(motor_positions, avg_sizes, c='blue', picker=5)
            self.ax.grid(True)

        self.canvas.draw()

    def analysis_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.motor_input.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_lbl.setText("Analysis finished.")

    def on_pick(self, event):
        if event.artist != self.scatter_plot:
            return

        ind = event.ind
        if len(ind) > 0:
            index = ind[0]
            # Retrieve motor position from the plot data
            offsets = self.scatter_plot.get_offsets()
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
                dialog = SecondaryPlotDialog(best_pos, data['bsrh_sizes'], data['bws_sizes'], data['bunches'], self)
                dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AnalysisGUI()
    gui.show()
    sys.exit(app.exec_())
