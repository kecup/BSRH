import sys

def main():
    with open("Analysis/AnalysisOfflineGUI.py", "r") as f:
        content = f.read()

    # Part 1: Add the button
    start_str_1 = """        view_layout.addLayout(plane_layout)
        exec_layout.addLayout(view_layout)

        self.btn_start_bsrh = QPushButton("Start BSRH Analysis")"""

    rep_1 = """        view_layout.addLayout(plane_layout)
        exec_layout.addLayout(view_layout)

        btn_action_layout = QHBoxLayout()
        self.btn_refine_mask = QPushButton("Refine Mask Position")
        self.btn_refine_mask.clicked.connect(self.refine_mask_position)
        btn_action_layout.addWidget(self.btn_refine_mask)

        self.btn_start_bsrh = QPushButton("Start BSRH Analysis")
        self.btn_start_bsrh.clicked.connect(self.start_bsrh_analysis)
        btn_action_layout.addWidget(self.btn_start_bsrh)

        exec_layout.addLayout(btn_action_layout)"""

    content = content.replace(start_str_1, rep_1)

    # Part 2: Add self.current_mask
    start_str_2 = """        self.worker = None
        self.canvas.mpl_connect('pick_event', self.on_pick)

    def add_directory(self):"""

    rep_2 = """        self.worker = None
        self.canvas.mpl_connect('pick_event', self.on_pick)

        self.current_mask = {'X': 455, 'Y': 300, 'R': 40}

    def add_directory(self):"""

    content = content.replace(start_str_2, rep_2)

    # Part 3: Replace start_bsrh_analysis entirely
    start_str_3 = """    def start_bsrh_analysis(self):"""
    end_str_3 = """        self.worker.start()"""
    start_idx = content.find(start_str_3)
    end_idx = content.find(end_str_3, start_idx) + len(end_str_3)

    rep_3 = """    def refine_mask_position(self):
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

        self.worker = AnalysisWorker('BSRH', directories=directories, params=params)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.status_signal.connect(self.status_lbl.setText)
        self.worker.error_signal.connect(self.handle_worker_error)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.start()"""

    content = content[:start_idx] + rep_3 + content[end_idx:]

    # Enable buttons on finish/error
    content = content.replace("        self.btn_start_bsrh.setEnabled(True)\n        if HAS_PYTIMBER: self.btn_fetch_bws.setEnabled(True)",
                              "        self.btn_start_bsrh.setEnabled(True)\n        self.btn_refine_mask.setEnabled(True)\n        if HAS_PYTIMBER: self.btn_fetch_bws.setEnabled(True)")

    with open("Analysis/AnalysisOfflineGUI.py", "w") as f:
        f.write(content)

if __name__ == "__main__":
    main()
