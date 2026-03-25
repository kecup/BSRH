import sys
import os

# Create a mock for dill/pickle and zstandard for the sandbox tests
# if they are not installed (like during automated CI tests).
try:
    import dill as pickle
except ImportError:
    import pickle
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import pyjapc
    HAS_PYJAPC = True
except ImportError:
    HAS_PYJAPC = False

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, QGroupBox,
    QFormLayout, QComboBox, QSplitter, QTreeWidget, QTreeWidgetItem,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Mock AllData class in case it's needed to unpickle files saved directly from Acquisition/GUI.py
class AllData:
    def __init__(self, timestamp, cam_data, device_data):
        self.timestamp = timestamp
        self.cam_data = cam_data
        self.device_data = device_data

# For compatibility with older files or pickle looking for Acquisition.GUI.AllData
import sys
import types
mock_module = types.ModuleType("GUI")
mock_module.AllData = AllData
sys.modules["GUI"] = mock_module

class DataBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Browser - BSRH/BWS Events")
        self.resize(1400, 800)

        self.all_events = [] # Store all loaded events: list of dicts/AllData objects
        self.filtered_events = [] # Store filtered events to display
        self.bg_image = None      # Store selected background image

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Splitter to divide the screen into 3 main columns: List/Filter, Details, Image
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- LEFT PANEL: File loading and Filtering ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Load Files Group
        load_group = QGroupBox("Load Data")
        load_layout = QVBoxLayout()
        self.btn_load = QPushButton("Load .pkl.zst Files")
        self.btn_load.clicked.connect(self.load_files)
        load_layout.addWidget(self.btn_load)

        self.btn_clear_data = QPushButton("Clear Data")
        self.btn_clear_data.clicked.connect(self.clear_all_data)
        load_layout.addWidget(self.btn_clear_data)

        self.lbl_loaded = QLabel("Loaded: 0 events")
        load_layout.addWidget(self.lbl_loaded)
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)

        # Filter Group
        filter_group = QGroupBox("Filter Events")
        filter_layout = QFormLayout()

        self.filter_key_input = QLineEdit()
        self.filter_key_input.setPlaceholderText("e.g. device_data['FBCT1']")

        self.filter_op_combo = QComboBox()
        self.filter_op_combo.addItems([">", "<", ">=", "<=", "==", "!="])

        self.filter_val_input = QLineEdit()
        self.filter_val_input.setPlaceholderText("e.g. 500")

        filter_layout.addRow("Key/Path:", self.filter_key_input)
        filter_layout.addRow("Operator:", self.filter_op_combo)
        filter_layout.addRow("Value:", self.filter_val_input)

        self.btn_apply_filter = QPushButton("Apply Filter")
        self.btn_apply_filter.clicked.connect(self.apply_filter)

        self.btn_clear_filter = QPushButton("Clear Filter")
        self.btn_clear_filter.clicked.connect(self.clear_filter)

        filter_layout.addRow(self.btn_apply_filter, self.btn_clear_filter)
        filter_group.setLayout(filter_layout)
        left_layout.addWidget(filter_group)

        # Event List
        self.event_list = QListWidget()
        self.event_list.itemSelectionChanged.connect(self.on_event_selected)
        left_layout.addWidget(QLabel("Events:"))
        left_layout.addWidget(self.event_list)

        # --- MIDDLE PANEL: Event Details (Tree) ---
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.addWidget(QLabel("Event Details:"))

        self.details_tree = QTreeWidget()
        self.details_tree.setHeaderLabels(["Key", "Value"])
        self.details_tree.setAlternatingRowColors(True)
        # Resize columns automatically
        self.details_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.details_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        middle_layout.addWidget(self.details_tree)

        # --- RIGHT PANEL: Image Visualization ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Image 2D Visualization:"))
        self.btn_set_bg = QPushButton("Set Current as Background")
        self.btn_set_bg.clicked.connect(self.set_current_as_bg)
        self.btn_clear_bg = QPushButton("Clear Background")
        self.btn_clear_bg.clicked.connect(self.clear_bg)
        bg_layout.addWidget(self.btn_set_bg)
        bg_layout.addWidget(self.btn_clear_bg)
        bg_layout.addStretch()
        right_layout.addLayout(bg_layout)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("cam_data['image2D']")
        self.ax.axis('off')
        self.img_plot = None

        # Image stats labels
        self.lbl_img_stats = QLabel("Image Stats: None")
        right_layout.addWidget(self.lbl_img_stats)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(middle_panel)
        splitter.addWidget(right_panel)

        # Set initial sizes (Left: 300, Middle: 400, Right: 700)
        splitter.setSizes([300, 400, 700])

    def load_files(self):
        if not HAS_ZSTD:
            QMessageBox.critical(self, "Missing Dependency", "The 'zstandard' module is required to decompress the .pkl.zst files. Please install it using 'pip install zstandard'.")
            return

        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Data Files", "", "Zstandard Pickles (*.pkl.zst);;All Files (*)"
        )

        if not file_paths:
            return

        dctx = zstd.ZstdDecompressor()
        new_events = 0

        for file_path in file_paths:
            try:
                with open(file_path, 'rb') as f:
                    with dctx.stream_reader(f) as reader:
                        data = pickle.load(reader)

                        # data could be a list of events or a single event
                        if isinstance(data, list):
                            self.all_events.extend(data)
                            new_events += len(data)
                        else:
                            self.all_events.append(data)
                            new_events += 1
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load {file_path}:\n{str(e)}")

        self.lbl_loaded.setText(f"Loaded: {len(self.all_events)} events")
        self.clear_filter() # Updates the list to show all events

    def clear_all_data(self):
        self.all_events.clear()
        self.filtered_events.clear()
        self.event_list.clear()
        self.details_tree.clear()
        self.ax.clear()
        self.ax.axis('off')
        self.canvas.draw()
        self.lbl_loaded.setText("Loaded: 0 events")
        self.lbl_img_stats.setText("Image Stats: None")
        self.filter_key_input.clear()
        self.filter_val_input.clear()
        self.clear_bg()

    def set_current_as_bg(self):
        selected_items = self.event_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        idx = item.data(Qt.UserRole)
        event = self.filtered_events[idx]

        if hasattr(event, "cam_data") and isinstance(event.cam_data, dict):
            img = event.cam_data.get('image2D')
            if img is not None and isinstance(img, np.ndarray) and len(img.shape) >= 2:
                if len(img.shape) == 3:
                    img = np.mean(img, axis=2)
                self.bg_image = img.astype(float).copy()
                self.plot_image(event)

    def clear_bg(self):
        self.bg_image = None
        selected_items = self.event_list.selectedItems()
        if selected_items:
            item = selected_items[0]
            idx = item.data(Qt.UserRole)
            event = self.filtered_events[idx]
            self.plot_image(event)

    def populate_event_list(self):
        self.event_list.clear()
        for i, event in enumerate(self.filtered_events):
            # Try to get a meaningful title, e.g. timestamp
            title = f"Event {i}"
            if hasattr(event, "timestamp"):
                title += f" - {event.timestamp}"

            item = QListWidgetItem(title)
            # Store the actual event object or its index in user data
            item.setData(Qt.UserRole, i)
            self.event_list.addItem(item)

    def apply_filter(self):
        key_path = self.filter_key_input.text().strip()
        op = self.filter_op_combo.currentText()
        val_str = self.filter_val_input.text().strip()

        if not key_path or not val_str:
            self.filtered_events = self.all_events.copy()
            self.populate_event_list()
            return

        try:
            target_val = float(val_str)
        except ValueError:
            target_val = val_str # fallback to string comparison if needed

        filtered = []
        for event in self.all_events:
            try:
                # Safely evaluate the key path against the event object.
                # E.g. key_path = "device_data['GATE-INT']['delaySlots']"
                # event is accessible as `event` in eval context.
                # To be safer, we can create a local dict.
                local_env = {"event": event}

                # If user didn't prepend "event.", let's try to handle standard properties:
                if key_path.startswith("device_data") or key_path.startswith("cam_data"):
                    eval_str = "event." + key_path
                elif key_path.startswith("event."):
                    eval_str = key_path
                else:
                    # Generic fallback
                    eval_str = "event." + key_path

                val = eval(eval_str, {"__builtins__": {}}, local_env)

                # If val is list/array, maybe they want the first item or sum?
                # For simplicity, we assume the path resolves to a scalar for comparison.
                # Convert target_val back to matching type if needed
                if isinstance(val, (int, float, np.number)):
                    t_val = float(target_val)
                else:
                    t_val = str(target_val)

                if op == ">": match = val > t_val
                elif op == "<": match = val < t_val
                elif op == ">=": match = val >= t_val
                elif op == "<=": match = val <= t_val
                elif op == "==": match = val == t_val
                elif op == "!=": match = val != t_val
                else: match = False

                if match:
                    filtered.append(event)
            except Exception as e:
                # If key doesn't exist or evaluation fails, we just skip it
                continue

        self.filtered_events = filtered
        self.populate_event_list()
        self.lbl_loaded.setText(f"Showing: {len(self.filtered_events)} / {len(self.all_events)} events")

    def clear_filter(self):
        self.filter_key_input.clear()
        self.filter_val_input.clear()
        self.filtered_events = self.all_events.copy()
        self.populate_event_list()
        self.lbl_loaded.setText(f"Showing: {len(self.filtered_events)} / {len(self.all_events)} events")

    def on_event_selected(self):
        selected_items = self.event_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        idx = item.data(Qt.UserRole)
        event = self.filtered_events[idx]

        self.populate_details(event)
        self.plot_image(event)

    def on_tree_item_double_clicked(self, item, column):
        # Walk up to the root to build the full key path
        path_parts = []
        current_item = item
        while current_item is not None:
            # Check if this node represents a list/array index (simple check by seeing if it's digit)
            # If so, it might need [] brackets instead of dict key access
            # For simplicity, we assume dict keys, e.g. "device_data['FBCT1']"
            part = current_item.text(0)
            path_parts.insert(0, part)
            current_item = current_item.parent()

        if not path_parts:
            return

        # Build the python expression string
        # root is usually 'device_data', 'cam_data', or 'timestamp'
        expr = path_parts[0]
        for p in path_parts[1:]:
            expr += f"['{p}']"

        self.filter_key_input.setText(expr)

    def populate_details(self, event):
        self.details_tree.clear()

        # Helper to recursively populate the tree
        def add_nodes(parent_item, data_dict):
            for key, val in data_dict.items():
                if isinstance(val, dict):
                    node = QTreeWidgetItem(parent_item, [str(key), ""])
                    add_nodes(node, val)
                elif isinstance(val, np.ndarray):
                    node = QTreeWidgetItem(parent_item, [str(key), f"Array {val.shape} {val.dtype}"])
                else:
                    node = QTreeWidgetItem(parent_item, [str(key), str(val)])

        # Root nodes
        if hasattr(event, "timestamp"):
            QTreeWidgetItem(self.details_tree, ["timestamp", str(event.timestamp)])

        if hasattr(event, "device_data") and isinstance(event.device_data, dict):
            dev_node = QTreeWidgetItem(self.details_tree, ["device_data", ""])
            add_nodes(dev_node, event.device_data)
            dev_node.setExpanded(True)

        if hasattr(event, "cam_data") and isinstance(event.cam_data, dict):
            cam_node = QTreeWidgetItem(self.details_tree, ["cam_data", ""])
            add_nodes(cam_node, event.cam_data)
            cam_node.setExpanded(True)

    def plot_image(self, event):
        self.ax.clear()
        self.img_plot = None
        self.lbl_img_stats.setText("Image Stats: None")

        if hasattr(event, "cam_data") and isinstance(event.cam_data, dict):
            img = event.cam_data.get('image2D')
            if img is not None and isinstance(img, np.ndarray):
                # Ensure 2D
                if len(img.shape) >= 2:
                    if len(img.shape) == 3:
                        # Convert to grayscale if RGB (just in case)
                        img = np.mean(img, axis=2)

                    img = img.astype(float)
                    title = "cam_data['image2D']"

                    if self.bg_image is not None and self.bg_image.shape == img.shape:
                        img = img - self.bg_image
                        title += " (Background Subtracted)"

                    self.img_plot = self.ax.imshow(img, cmap='viridis', aspect='equal', interpolation='nearest')
                    self.ax.set_title(title)

                    # Update stats
                    stats_text = f"Stats: Min={np.min(img):.2f}, Max={np.max(img):.2f}, Mean={np.mean(img):.2f}, Sum={np.sum(img):.2f}"
                    self.lbl_img_stats.setText(stats_text)
                else:
                    self.ax.set_title("Image is 1D (Projections)")
                    self.ax.plot(img)
            else:
                self.ax.set_title("No image2D data found")
        else:
             self.ax.set_title("No cam_data found")

        self.canvas.draw()


if __name__ == "__main__":
    if HAS_PYJAPC:
        try:
            japc = pyjapc.PyJapc()
        except Exception as e:
            print(f"Warning: Failed to initialize pyjapc: {e}")

    app = QApplication(sys.argv)
    window = DataBrowser()
    window.show()
    sys.exit(app.exec_())
