import os
import sys
import threading
import time
import math
import zstandard as zstd
import dill as pickle
import numpy as np
import datetime as dt
from scipy.optimize import curve_fit, minimize
from scipy import stats

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QProgressBar, QMessageBox, QDialog,
    QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Utility functions for fitting and convolution
def gaussian(x, amp, cen, wid):
    x = np.asarray(x)
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def fit_gaussian(x, y, p0):
    try:
        popt, pcov = curve_fit(gaussian, x, y, p0=p0, maxfev=2000)
        return popt
    except Exception:
        return None

def chi2(a, b):
    if len(a) != len(b):
        raise ValueError("Lengths of a and b must match")
    a = np.asarray(a)
    b = np.asarray(b)
    resid2_mean = np.mean((a - b) ** 2)
    return 0.5 * len(a) * np.log(resid2_mean)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] -= ret[:-n]
    result = ret[n - 1:] / n
    pad = [result[0]] * (n - 1)
    return np.array(pad + list(result))

def objFunc(params, bwsArray, bsrhArray, fbctArray, plane):
    # params: [PSF, Pixel Size, Amplitude Scale, ... ]
    # plane: 'H' or 'V'
    newX = np.linspace(-4, 4, 1000)
    tmpPSF = gaussian(newX, 1, 0, params[0])
    tmpPSF = tmpPSF / np.sum(tmpPSF)

    n = len(bwsArray)
    totalSquares = 0
    for bunch in range(n):
        # Scale BSRH by Pixel Size
        bsrh = np.interp(newX, params[1] / 1000 * bsrhArray[bunch][0], bsrhArray[bunch][1])

        # Convolve BWS with PSF
        bws = np.convolve(bwsArray[bunch], tmpPSF, 'same')

        # Determine edges to prevent boundary effects
        ll = 499
        rl = 501

        if plane == 'H':
            while ll > 0 and bws[ll+1] > bws[ll]:
                ll -= 1
            while rl < len(bws) and bws[rl-1] < bws[rl]:
                rl += 1
            crop = 510
        else: # 'V'
            while ll > 0 and bws[ll + 1] > bws[ll]:
                ll -= 1
            while rl < len(bws) and bws[rl - 1] > bws[rl]:
                rl += 1
            crop = -1

        bsrh_sum = np.sum(bsrh[:crop])
        if bsrh_sum > 0:
            bsrh = np.sum(bws[:crop]) / bsrh_sum * bsrh

        # Calculate log-residual cost using cropped vectors
        if rl > ll:
            totalSquares += chi2(params[2] * bsrh[ll:rl], bws[ll:rl])

    return totalSquares

# Worker thread for analyzing data
class AnalysisThread(QThread):
    progress_signal = pyqtSignal(int, int) # Current file, total files
    update_plot_signal = pyqtSignal(dict) # Emits the grouped data
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, bsrh_path, bws_path, motor_name, plane):
        super().__init__()
        self.bsrh_path = bsrh_path
        self.bws_path = bws_path
        self.motor_name = motor_name
        self.plane = plane # 'H' or 'V'
        self.running = True

        self.processed_bws = set()
        self.processed_bsrh = set()

        self.bws_data = [] # Stores full profiles per timestamp
        self.bsrh_grouped_data = {} # Keyed by motor_pos

        self.dctx = zstd.ZstdDecompressor()

    def stop(self):
        self.running = False

    def run(self):
        self.status_signal.emit("Started real-time analysis...")
        while self.running:
            bws_updated = self.process_new_bws(self.bws_path)
            bsrh_updated = self.process_new_bsrh(self.bsrh_path)

            if bws_updated or bsrh_updated:
                # Group data and attempt rigorous minimization
                self.perform_optimization()
                self.update_plot_signal.emit(self.bsrh_grouped_data)

            time.sleep(1.0) # Poll every 1 second

        self.status_signal.emit("Analysis stopped.")
        self.finished_signal.emit()

    def process_new_bws(self, directory):
        updated = False
        if not os.path.isdir(directory):
            self.status_signal.emit(f"BWS directory not found: {directory}")
            return updated

        files = sorted([f for f in os.listdir(directory) if f.endswith('.pkl.zst') and f not in self.processed_bws])
        if not files: return updated

        for filename in files:
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
                bunch_list = data.value['bunchList']

                # Assume scanning from -4 to 4 mm
                newX = np.linspace(-4, 4, 1000)

                bunches_profiles = {}
                max_bunch = int(len(proj_in) / pts_in)

                ratio = 1.023 if self.plane == 'H' else 0.936

                for k in range(min(max_bunch, len(bunch_list))):
                    bunch_num = bunch_list[k]

                    # IN scan
                    y_in = proj_in[k * pts_in:(k + 1) * pts_in]
                    x_in = pos_in[:pts_in]
                    y_in = y_in - np.mean(y_in[:50])
                    y_in = moving_average(np.asarray(y_in), 1)

                    # Quick fit to center profile
                    popt_in = fit_gaussian(x_in, y_in, [max(y_in), np.mean(x_in), 0.28])

                    if popt_in is not None:
                        x_in_shifted = (x_in - popt_in[1]) * ratio
                        y_in_interp = np.interp(newX, x_in_shifted, y_in)
                        bunches_profiles[bunch_num] = y_in_interp

                self.bws_data.append({
                    'timestamp': timestamp,
                    'bunches': bunches_profiles
                })
                self.processed_bws.add(filename)
                updated = True
            except Exception as e:
                print(f"Error processing BWS file {filename}: {e}")

        self.bws_data.sort(key=lambda x: x['timestamp'])
        return updated

    def process_new_bsrh(self, directory):
        updated = False
        if not os.path.isdir(directory):
            self.status_signal.emit(f"BSRH directory not found: {directory}")
            return updated

        files = sorted([f for f in os.listdir(directory) if f.endswith('.pkl.zst') and f not in self.processed_bsrh])
        if not files: return updated

        for filename in files:
            if not self.running: break
            self.status_signal.emit(f"Processing new BSRH file: {filename}")

            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'rb') as f:
                    with self.dctx.stream_reader(f) as reader:
                        bsrh_events = pickle.load(reader)

                for event in bsrh_events:
                    timestamp = dt.datetime.fromtimestamp(event.cam_data['imageTimeStamp']/1e9)

                    motor_pos = None
                    if self.motor_name in event.device_data:
                        motor_data = event.device_data[self.motor_name]
                        if isinstance(motor_data, dict) and 'position' in motor_data:
                            motor_pos = motor_data['position']
                        else:
                            motor_pos = motor_data

                    if motor_pos is None: continue

                    bunch_slot = event.device_data.get('GATE-INT', {}).get('delaySlots', None)
                    if bunch_slot is None: continue

                    closest_bws_profile = self.find_closest_bws(timestamp, bunch_slot)
                    if closest_bws_profile is None: continue

                    # Process BSRH profile
                    image = event.cam_data['image2D']

                    if self.plane == 'H':
                        prof = np.sum(image[154:454, :], axis=0)
                    else: # V
                        prof = np.sum(image[:, 284:684], axis=1)

                    prof = prof - np.mean(prof[:50])

                    y = np.asarray(prof)
                    max_index = np.argmax(y)
                    x = np.arange(len(y))

                    if self.plane == 'H':
                        popt = fit_gaussian(x[:max_index+30], y[:max_index+30], [np.max(y), max_index, 25.0])
                    else:
                        popt = fit_gaussian(x, y, [np.max(y), max_index, 25.0])

                    if popt is not None:
                        x = x - popt[1]

                        fbct_val = 1.0
                        if 'FBCT1' in event.device_data:
                            bStart = bunch_slot
                            bStop = bunch_slot + int(event.device_data['GATE-INT'].get('numberOfPulses', 4) / 4.0)
                            fbct_val = np.sum(event.device_data['FBCT1'].get('averageBunchIntensities', [1.0])[bStart:bStop])

                        if motor_pos not in self.bsrh_grouped_data:
                            self.bsrh_grouped_data[motor_pos] = {
                                'bsrh_profiles': [], 'bws_profiles': [], 'fbct': [], 'bunches': [],
                                'optimized': False, 'psf': None, 'pixel_size': None
                            }

                        self.bsrh_grouped_data[motor_pos]['bsrh_profiles'].append([x, y])
                        self.bsrh_grouped_data[motor_pos]['bws_profiles'].append(closest_bws_profile)
                        self.bsrh_grouped_data[motor_pos]['fbct'].append(fbct_val)
                        self.bsrh_grouped_data[motor_pos]['bunches'].append(bunch_slot)
                        self.bsrh_grouped_data[motor_pos]['optimized'] = False # Re-flag for optimization

                self.processed_bsrh.add(filename)
                updated = True

            except Exception as e:
                print(f"Error processing BSRH file {filename}: {e}")

        return updated

    def find_closest_bws(self, bsrh_time, bunch_slot):
        best_diff = float('inf')
        best_profile = None

        for bws in self.bws_data:
            if bunch_slot in bws['bunches']:
                diff = abs((bsrh_time - bws['timestamp']).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_profile = bws['bunches'][bunch_slot]

        return best_profile

    def perform_optimization(self):
        for pos, data in self.bsrh_grouped_data.items():
            if not data['optimized'] and len(data['bws_profiles']) > 0:
                self.status_signal.emit(f"Running optimization for motor pos {pos}...")

                # Initial Guesses and Bounds
                if self.plane == 'H':
                    initGuess = [0.6, 20.0, 1.0, 1]
                    bnds = ((0.4, 0.7), (16, 24), (0.1, 2.0), (0, 5))
                else:
                    initGuess = [0.25, 18.0, 1, 1]
                    bnds = ((0.2, 0.4), (15, 22), (0.1, 2.0), (0, 5))

                try:
                    result = minimize(objFunc, initGuess,
                                      args=(data['bws_profiles'], data['bsrh_profiles'], data['fbct'], self.plane),
                                      bounds=bnds, method='TNC')
                    out = result.x

                    data['psf'] = out[0]
                    data['pixel_size'] = out[1]
                    data['optimized'] = True
                    self.status_signal.emit(f"Optimized pos {pos}: PSF={out[0]:.3f}, PxSize={out[1]:.2f}")
                except Exception as e:
                    print(f"Optimization failed for pos {pos}: {e}")

class SecondaryPlotDialog(QDialog):
    def __init__(self, motor_pos, data, plane, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Convolved Match for Motor Position: {motor_pos}")
        self.resize(800, 600)
        self.plane = plane

        self.layout = QVBoxLayout(self)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.plot_data(data)

    def plot_data(self, data):
        if not data['optimized']:
            return

        ax = self.figure.add_subplot(111)
        ax.clear()

        # Just plot the first bunch as an example of the match
        bunch_idx = 0

        psf_val = data['psf']
        px_size = data['pixel_size']

        newX = np.linspace(-4, 4, 1000)
        tmpPSF = gaussian(newX, 1, 0, psf_val)
        tmpPSF = tmpPSF / np.sum(tmpPSF)

        bsrh_x, bsrh_y = data['bsrh_profiles'][bunch_idx]
        bws_y = data['bws_profiles'][bunch_idx]
        fbct = data['fbct'][bunch_idx]

        bsrh = np.interp(newX, px_size / 1000 * bsrh_x, bsrh_y)
        bsrh_sum = np.sum(bsrh)
        if bsrh_sum > 0:
            bsrh = fbct / bsrh_sum * bsrh

        bws = np.convolve(bws_y, tmpPSF, 'same')
        bws_sum = np.sum(bws)
        if bws_sum > 0:
            bws = fbct / bws_sum * bws

        ax.plot(newX, bsrh, label="BSRH (Scaled)")
        ax.plot(newX, bws, label="BWS (Convolved)")

        info_text = (f"Pixel Size: {px_size:.2f} μm/px\n"
                     f"Magnification: {1000/px_size:.2f}\n"
                     f"PSF: {psf_val*1000:.2f} μm")

        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Intensity')
        ax.set_title(f"BSRH and BWS Convolution Match (Bunch {data['bunches'][bunch_idx]})")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

class AnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSRH/BWS PSF & Magnification Calibration")
        self.resize(1000, 700)

        self.layout = QVBoxLayout(self)

        input_layout = QHBoxLayout()

        self.bsrh_btn = QPushButton("Select BSRH Dir")
        self.bsrh_btn.clicked.connect(self.select_bsrh_dir)
        self.bsrh_lbl = QLabel("No directory selected")

        self.bws_btn = QPushButton("Select BWS Dir")
        self.bws_btn.clicked.connect(self.select_bws_dir)
        self.bws_lbl = QLabel("No directory selected")

        self.motor_input = QLineEdit()
        self.motor_input.setPlaceholderText("Enter motor name (e.g., M3-X)")

        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["H", "V"])

        input_layout.addWidget(self.bsrh_btn)
        input_layout.addWidget(self.bsrh_lbl)
        input_layout.addWidget(self.bws_btn)
        input_layout.addWidget(self.bws_lbl)
        input_layout.addWidget(QLabel("Plane:"))
        input_layout.addWidget(self.plane_combo)
        input_layout.addWidget(QLabel("Motor:"))
        input_layout.addWidget(self.motor_input)

        self.layout.addLayout(input_layout)

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

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax1 = self.figure.add_subplot(211)
        self.ax1.set_xlabel("Motor Position")
        self.ax1.set_ylabel("Pixel Size (μm/px)")
        self.ax1.set_title("Pixel Size vs Motor Position")

        self.ax2 = self.figure.add_subplot(212)
        self.ax2.set_xlabel("Motor Position")
        self.ax2.set_ylabel("PSF (μm)")
        self.ax2.set_title("PSF vs Motor Position")

        self.figure.tight_layout()
        self.canvas.draw()

        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.bsrh_path = ""
        self.bws_path = ""
        self.worker = None
        self.grouped_data = {}
        self.scatter_plot1 = None
        self.scatter_plot2 = None

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
        plane = self.plane_combo.currentText()
        if not self.bsrh_path or not self.bws_path or not motor_name:
            QMessageBox.warning(self, "Input Error", "Please select directories and provide a motor name.")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.motor_input.setEnabled(False)
        self.plane_combo.setEnabled(False)

        self.grouped_data.clear()
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_ylabel("Pixel Size (μm/px)")
        self.ax2.set_ylabel("PSF (μm)")
        self.canvas.draw()

        self.worker = AnalysisThread(self.bsrh_path, self.bws_path, motor_name, plane)
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
        px_sizes = []
        psfs = []

        for pos, data in grouped_data.items():
            if data['optimized']:
                motor_positions.append(pos)
                px_sizes.append(data['pixel_size'])
                psfs.append(data['psf'] * 1000.0) # Convert to um

        self.ax1.clear()
        self.ax2.clear()

        motor_name = self.motor_input.text().strip()
        self.ax1.set_xlabel(f"Motor Position ({motor_name})")
        self.ax1.set_ylabel("Pixel Size (μm/px)")
        self.ax1.set_title("Pixel Size vs Motor Position")

        self.ax2.set_xlabel(f"Motor Position ({motor_name})")
        self.ax2.set_ylabel("PSF (μm)")
        self.ax2.set_title("PSF vs Motor Position")

        if motor_positions:
            self.scatter_plot1 = self.ax1.scatter(motor_positions, px_sizes, c='blue', picker=5)
            self.scatter_plot2 = self.ax2.scatter(motor_positions, psfs, c='red', picker=5)
            self.ax1.grid(True)
            self.ax2.grid(True)

        self.canvas.draw()

    def analysis_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.motor_input.setEnabled(True)
        self.plane_combo.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_lbl.setText("Analysis finished.")

    def on_pick(self, event):
        if event.artist not in [self.scatter_plot1, self.scatter_plot2]:
            return

        ind = event.ind
        if len(ind) > 0:
            index = ind[0]
            offsets = event.artist.get_offsets()
            motor_pos = offsets[index][0]

            best_pos = None
            min_diff = float('inf')
            for pos in self.grouped_data.keys():
                if not self.grouped_data[pos]['optimized']: continue
                diff = abs(pos - motor_pos)
                if diff < min_diff:
                    min_diff = diff
                    best_pos = pos

            if best_pos is not None:
                data = self.grouped_data[best_pos]
                plane = self.plane_combo.currentText()
                dialog = SecondaryPlotDialog(best_pos, data, plane, self)
                dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AnalysisGUI()
    gui.show()
    sys.exit(app.exec_())
