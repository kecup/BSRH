# Data acquisition via master camera and parameter subscriptions
# Now subscribe to all devices, cache their latest values, and use camera callback to assemble full events.

import pyjapc
import pyrbac
import time
import sys
import os
import threading
import dill as pickle
import zstandard as zstd
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QSpinBox, QLabel, QHBoxLayout, QTreeWidget, QTreeWidgetItem, QMessageBox, QLineEdit)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt
from queue import Queue

# Configuration
DEVICE_LIST = {
    'L4': 'LHC.BSRH.L4-Z-TS50.B2/Acquisition#position',
    'L5z': 'LHC.BSRH.L5-Z-TS25.B2/Acquisition#position',
    'L5x':'LHC.BSRH.L5-X-TS25.B2/Acquisition#position',
    'Folding-V':'LHC.BSRH.FOLDING-V.B2/Acquisition#position',
    'Folding-H':'LHC.BSRH.FOLDING-H.B2/Acquisition#position',
    'Collimator-V':'TCP.D6R7.B2/MeasuredCornerPositions',
    'Collimator-H':'TCP.C6R7.B2/MeasuredCornerPositions',
    'FBCT1':'LHC.BCTFR.A6R4.B2/Acquisition',
    'FBCT2':'LHC.BCTFR.B6R4.B2/Acquisition',
    'CM-FWL':'LHC.BSRH.CM-FWL.B2/Acquisition#position', 
    'CMASK-X':'LHC.BSRH.CMASK-X-TSZ100.B2/Acquisition#position',
    'CMASK-Y':'LHC.BSRH.CMASK-Y-TSZ100.B2/Acquisition#position',
    'EMASK-HL':'LHC.BSRH.EMASK-H-LEFT.B2/Acquisition#position',
    'EMASK-HR':'LHC.BSRH.EMASK-H-RIGHT.B2/Acquisition#position',
    'EMASK-VD':'LHC.BSRH.EMASK-V-DOWN.B2/Acquisition#position',
    'EMASK-VU':'LHC.BSRH.EMASK-V-UP.B2/Acquisition#position',
    'FW-COLOR':'LHC.BSRH.FW-COLOR.B2/Acquisition#position',
    'LYOT-HL':'LHC.BSRH.LYOT-H-LEFT.B2/Acquisition#position',
    'LYOT-HR':'LHC.BSRH.LYOT-H-RIGHT.B2/Acquisition#position',
    'LYOT-VU':'LHC.BSRH.LYOT-V-UP.B2/Acquisition#position',
    'LYOT-VD':'LHC.BSRH.LYOT-V-DOWN.B2/Acquisition#position',
    'M1-X':'LHC.BSRH.M1-X-TS50.B2/Acquisition#position',
    'M3-PHI':'LHC.BSRH.M3-PHI.B2/Acquisition#position',
    'M3-THETA':'LHC.BSRH.M3-THETA.B2/Acquisition#position',
    'M3-X':'LHC.BSRH.M3-X-TSZ100.B2/Acquisition#position',
    'M3-Y':'LHC.BSRH.M3-Y-TSZ100.B2/Acquisition#position',
    'M3-Z':'LHC.BSRH.M3-Z-TSZ100.B2/Acquisition#position',
    'STEERING-H':'LHC.BSRH.STEERING-H.B2/Acquisition#position',
    'STEERING-V':'LHC.BSRH.STEERING-V.B2/Acquisition#position',
    'INTENSIFIER':'LHC.BSRH.IMGINT.5L4.B2/Acquisition',
    'GATE-CAM':'LHC.BSRH.GATE-GEN.5L4.B2/Setting2',
    'FILTERS.B2':'LHC.BSRH.FILTERS.B2/Acquisition'
}
CAMERA_DEVICE = 'LHC.BSRH.BTVDC-CAM.MAIN/LastImage'
INTENSIFIER_DEVICE = 'LHC.BSRH.GATE-GEN.5L4.B2/Setting1'
DATA_PATH_ROOT = '/user/jpucek/BSRH/Data'
FILE_SIZE = 100

# Globals
cctx = zstd.ZstdCompressor(level=5)
data_cache = {}             # stores latest values for each parameter
cache_lock = threading.Lock()
event_queue = Queue()       # to queue complete events for background saving
saver_running = threading.Event()
setter_running = threading.Event()
camera_arrived = threading.Event()
firstCallback = True
scanning = False
run_number = 0


class AllData:
    def __init__(self, timestamp, cam_data, device_data):
        self.timestamp = timestamp
        self.cam_data = cam_data
        self.device_data = device_data

# Background saver thread: takes events from queue and writes them in chunks

def saver_worker():
    buffer = []
    print("[Saver] Worker started.")
    while saver_running.is_set() or not event_queue.empty():
        try:
            ev = event_queue.get(timeout=0.5)
            buffer.append(ev)
            if len(buffer) >= FILE_SIZE:
                write_chunk(buffer)
                buffer.clear()
        except:
            continue
    if buffer:
        write_chunk(buffer)
    print("[Saver] Worker exiting.")
    
    
def setter_worker(japc, path, value, freq):
    global scanning
    print("[Setter] Worker started.")
    japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting2#triggerDecimation', '0')
    interval = 1.0 / freq-0.005  #compensating for the set time
    while setter_running.is_set():
      if scanning:
        if not camera_arrived.wait(timeout = 2):
          continue
        
        try:
          japc.setParam(path,value)
          
        except Exception as e:
            print(f"[Setter] Error setting parameter: {e}")
        
        camera_arrived.clear()
      
      else:
        try:
            japc.setParam(path, value)
            # Synchronization point: can add checks against saver_running or other events here
            time.sleep(interval)
        except Exception as e:
            print(f"[Setter] Error setting parameter: {e}")
    japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting2#triggerDecimation', '11245')
    print("[Setter] Worker exiting.")


def write_chunk(events):
    ts = time.strftime('%Y_%m_%d', time.localtime())
    filename = f"{ts}_run_{run_number:02d}_chunk_{int(time.time())}.pkl.zst"
    path = os.path.join(DATA_PATH_ROOT, time.strftime('%Y/%m/%d/', time.localtime()))
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    with open(full_path, 'wb') as f:
      with cctx.stream_writer(f) as compressor:
        pickle.dump(events, compressor)
    print(f"Wrote {len(events)} events to {full_path}")

# Utility
def get_last_run_number():
    today_dir = os.path.join(DATA_PATH_ROOT, time.strftime('%Y/%m/%d/', time.localtime()))
    if not os.path.isdir(today_dir): return 0
    files = os.listdir(today_dir)
    runs = []
    for f in files:
        parts = f.split('_run_')
        if len(parts) > 1:
            num = parts[1].split('_')[0]
            if num.isdigit(): runs.append(int(num))
    return max(runs) if runs else 0
 
    
# GUI Definition
class AcquisitionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Acquisition GUI")
        self.setFixedSize(350, 300)
        self.main_layout = QVBoxLayout()
        
        # Controls layout
        controls_layout = QHBoxLayout()
        self.freq_label = QLabel("Freq(Hz):")
        self.freq_input = QSpinBox()
        self.freq_input.setRange(1, 5)
        self.freq_input.setValue(1)
        controls_layout.addWidget(self.freq_label)
        controls_layout.addWidget(self.freq_input)
        
        self.setter_btn = QPushButton("Start Setter")
        self.setter_btn.clicked.connect(self.toggle_setter)
        controls_layout.addWidget(self.setter_btn)

        self.acq_btn = QPushButton("Start Acquisition")
        self.acq_btn.clicked.connect(self.toggle_acquisition)
        controls_layout.addWidget(self.acq_btn)

        self.main_layout.addLayout(controls_layout)
        self.setLayout(self.main_layout)
        

        # getBunches button
        self.get_bunches_btn = QPushButton("Get Bunches")
        self.get_bunches_btn.clicked.connect(self.get_bunches)
        self.main_layout.addWidget(self.get_bunches_btn)

        # Tree for bunch selection
        self.bunch_tree = QTreeWidget()
        self.bunch_tree.setHeaderLabels(["Bunch ID", "Scan"])
        self.main_layout.addWidget(self.bunch_tree)

        self.new_bunch_input = QLineEdit()
        self.new_bunch_input.setPlaceholderText("Enter bunch ID (0-3550)")
        validator = QIntValidator(0, 3550)
        self.new_bunch_input.setValidator(validator)
        self.add_bunch_btn = QPushButton("Add Bunch")
        self.add_bunch_btn.clicked.connect(self.add_bunch)
        h_manual = QHBoxLayout()
        h_manual.addWidget(self.new_bunch_input)
        h_manual.addWidget(self.add_bunch_btn)
        self.main_layout.addLayout(h_manual)
        
        self.scan_bunches_btn = QPushButton("Scan bunches")
        self.scan_bunches_btn.clicked.connect(self.scan_bunches)
        self.main_layout.addWidget(self.scan_bunches_btn)
        
        self.acquiring = False
        self.setting = False
        self.saver_thread = None
        self.setter_thread = None
        
        self.bunchSlots = []
        self.bunchIndex = 0
        
        self.japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
        self.clientRBAC = pyrbac.AuthenticationClient()
        self.token = self.clientRBAC.login_kerberos()
        self.japc.rbacLogin(token = self.token)
        # Subscribe to all device parameters
        for alias, path in DEVICE_LIST.items():
            self.japc.subscribeParam(path, lambda name, value, a=alias: self.generic_callback(a, value))
        # Subscribe to camera with header
        self.japc.subscribeParam(CAMERA_DEVICE, self.camera_callback, getHeader=True)
        self.japc.subscribeParam(INTENSIFIER_DEVICE, self.intensifier_callback, getHeader = False)
        
        
   # Subscription callbacks

    def generic_callback(self, name, value):
        # store latest value in cache
        with cache_lock:
            data_cache[name] = value

    def intensifier_callback(self, name, value, header=None):
        if scanning:
          ##Here sending trigger
          camera_arrived.set()
        with cache_lock:
            data_cache['GATE-INT'] = value

    def camera_callback(self, name, image, header):
        global firstCallback, scanning
                
        if firstCallback:
          firstCallback = False
          return
        # on camera trigger, assemble event from cache snapshot plus image
        with cache_lock:
            snapshot = data_cache.copy()
        if scanning:
          if self.bunchIndex+1 < len(self.bunchSlots):
            self.bunchIndex = self.bunchIndex + 1
          else:
            self.bunchIndex = 0
          self.japc.setParam('LHC.BSRH.GATE-GEN.5L4.B2/Setting1#delaySlots', self.bunchSlots[self.bunchIndex])
        #camera_arrived.set()
        timestamp = header.get('acqStamp', time.time())
        ev = AllData(timestamp, image, snapshot)
        event_queue.put(ev)
        print(f"Queued event at {timestamp}")
        
    def get_bunches(self):
        # Populate tree with dummy bunches and checkboxes
        self.bunch_tree.clear()
        bunchIntensities = self.japc.getParam('LHC.BCTFR.A6R4.B2/Acquisition#averageBunchIntensities')
        for i in range(len(bunchIntensities)):
          if bunchIntensities[i] > 0:
            #self.bunchSlots.append(i)
            item = QTreeWidgetItem([str(i), ""])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(1, Qt.Checked)
            self.bunch_tree.addTopLevelItem(item)
            
            
    def add_bunch(self):
        text = self.new_bunch_input.text().strip()
        if text == "":
            return
        item = QTreeWidgetItem([text, ""])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked)
        self.bunch_tree.addTopLevelItem(item)
        self.new_bunch_input.clear()

    def get_checked_bunches(self):
      checked = []
      root = self.bunch_tree.invisibleRootItem()
      for i in range(root.childCount()):
          item = root.child(i)
          if item.checkState(1) == Qt.Checked:
              try:
                  checked.append(int(item.text(0)))
              except ValueError:
                  continue
      self.bunchSlots = checked
      return checked

    def scan_bunches(self):
        global scanning
        
        if not scanning:
          print(self.get_checked_bunches())
          if self.bunchSlots == []:
            print('Bunches are empty!')
            return
          
          #Setter and saver must be running for the bunch scan to work - maybe we could use the subscriptions without saving the data...
          if setter_running.is_set() == False:
            self.toggle_setter()
          if saver_running.is_set() == False:
            self.toggle_acquisition()
            
          
          self.setter_btn.setEnabled(False)
          self.acq_btn.setEnabled(False)
          scanning = True
          self.scan_bunches_btn.setText("Stop bunch scan")
          self.freq_input.setReadOnly(True)
          camera_arrived.set()
          
        else:
          scanning = False
          self.freq_input.setReadOnly(False)
          self.setter_btn.setEnabled(True)
          self.acq_btn.setEnabled(True)
          self.scan_bunches_btn.setText("Scan bunches")
        
    def toggle_acquisition(self):
        global saver_running, run_number, firstCallback
        if not self.acquiring:
            try:
                # Start subscriptions and saver thread
                run_number = get_last_run_number() + 1
                self.japc.startSubscriptions()
                saver_running.set()
                self.saver_thread = threading.Thread(target=saver_worker, daemon=True)
                self.saver_thread.start()
                self.acq_btn.setText("Stop Acquisition")
                self.acquiring = True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start:\n{e}")
        else:
            try:
                # Stop subscriptions and saver thread
                self.japc.stopSubscriptions()
                firstCallback = True
                saver_running.clear()
                if self.saver_thread:
                    self.saver_thread.join(timeout=10)
                self.acq_btn.setText("Start Acquisition")
                self.acquiring = False
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to stop:\n{e}")
    
                
    def toggle_setter(self):
        if not self.setting:
            try:
                freq = self.freq_input.value()
                setter_running.set()
                setter_path = 'LHC.BSRH.GATE-GEN.5L4.B2/Trig#triggerSW'
                setter_value = 2
                self.setter_thread = threading.Thread(
                    target=setter_worker,
                    args=(self.japc, setter_path, setter_value, freq),
                    daemon=True
                )
                self.setter_thread.start()
                self.setter_btn.setText("Stop Setter")
                self.setting = True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start setter:\n{e}")
        else:
            try:
                setter_running.clear()
                if self.setter_thread:
                    self.setter_thread.join(timeout=2)
                self.setter_btn.setText("Start Setter")
                self.setting = False
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to stop setter:{e}")
    
                
    def closeEvent(self, event):
        # Stop acquisition if running
        if self.acquiring:
            try:
                self.japc.stopSubscriptions()
                saver_running.clear()
                if self.saver_thread:
                    self.saver_thread.join(timeout=5)
            except:
                pass
        # Stop setter if running
        if self.setting:
            try:
                setter_running.clear()
                if self.setter_thread:
                    self.setter_thread.join(timeout=2)
            except:
                pass
        event.accept()

# Entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AcquisitionGUI()
    gui.show()
    sys.exit(app.exec_())

