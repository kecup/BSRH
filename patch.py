import sys

def main():
    with open("Analysis/AnalysisOfflineGUI.py", "r") as f:
        content = f.read()

    start_str = "    def start_bsrh_analysis(self):"
    end_str = "        self.update_plot()"

    start_idx = content.find(start_str)
    # find the end_str *after* start_idx
    end_idx = content.find(end_str, start_idx) + len(end_str)

    if start_idx == -1 or end_idx == -1:
        print("Could not find block")
        return

    replacement = """    def start_bsrh_analysis(self):
        if len(self.dir_widgets) == 0:
            QMessageBox.warning(self, "No Directories", "Please add at least one directory.")
            return

        params = self.get_params()
        if not params:
            return

        directories = [widget.get_range() for widget in self.dir_widgets]

        # Determine the mask from the first valid masked image in the selection
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

        if len(all_files) == 0:
            QMessageBox.warning(self, "No files", "No data to process.")
            return

        maskX, maskY, maskR = 455, 300, 40 # defaults
        found_masked = False

        for filepath in all_files:
            if found_masked: break
            try:
                with open(filepath, 'rb') as f:
                    with dctx.stream_reader(f) as reader:
                        file_data = pickle.load(reader)
                        for event in file_data:
                            cmask_x = event.device_data.get('CMASK-X', None)
                            if cmask_x is not None:
                                cmask_x_val = np.asarray(cmask_x).item() if isinstance(cmask_x, np.ndarray) else cmask_x
                                if abs(cmask_x_val - 455) < 100:
                                    image = event.cam_data['image2D']
                                    if image.shape[0] <= 2:
                                        found_masked = True
                                        break

                                    center_region = image[154:454, 284:684]
                                    min_idx = np.unravel_index(np.argmin(center_region, axis=None), center_region.shape)
                                    guessedY = min_idx[0] + 154
                                    guessedX = min_idx[1] + 284

                                    dialog = MaskValidationDialog(image, guessedX, guessedY, maskR, self)
                                    if dialog.exec_() == QDialog.Accepted:
                                        maskX, maskY, maskR = dialog.get_mask_params()
                                        found_masked = True
                                    else:
                                        self.status_lbl.setText("Analysis aborted (mask validation cancelled).")
                                        return
                                    break
            except Exception as e:
                pass

        if not found_masked:
            QMessageBox.warning(self, "Warning", "Could not find a masked image to validate the mask. Using defaults.")

        params['maskX'] = maskX
        params['maskY'] = maskY
        params['maskR'] = maskR

        self.btn_start_bsrh.setEnabled(False)
        self.btn_fetch_bws.setEnabled(False)
        self.status_lbl.setText("Analyzing BSRH...")

        self.worker = AnalysisWorker('BSRH', directories=directories, params=params)
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

            self.worker = AnalysisWorker('BWS', bws_args=(t1, t2, bunches_to_fetch), params=params)
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
        if HAS_PYTIMBER: self.btn_fetch_bws.setEnabled(True)

    def analysis_finished(self, new_results):
        # Update self.results with the new data
        for plane in ['H', 'V']:
            # For BSRH
            for bunch, data in new_results[plane]['bsrh'].items():
                if bunch not in self.results[plane]['bsrh']:
                    self.results[plane]['bsrh'][bunch] = {'time': [], 'rpi': [], 'is_bg': []}
                self.results[plane]['bsrh'][bunch]['time'].extend(data['time'])
                self.results[plane]['bsrh'][bunch]['rpi'].extend(data['rpi'])
                self.results[plane]['bsrh'][bunch]['is_bg'].extend(data['is_bg'])
            # For BWS
            for bunch, data in new_results[plane]['bws'].items():
                if bunch not in self.results[plane]['bws']:
                    self.results[plane]['bws'][bunch] = {'time': [], 'rpi': []}
                self.results[plane]['bws'][bunch]['time'].extend(data['time'])
                self.results[plane]['bws'][bunch]['rpi'].extend(data['rpi'])

        self.btn_start_bsrh.setEnabled(True)
        if HAS_PYTIMBER: self.btn_fetch_bws.setEnabled(True)
        self.update_bunch_list()
        self.update_plot()"""

    new_content = content[:start_idx] + replacement + content[end_idx:]

    with open("Analysis/AnalysisOfflineGUI.py", "w") as f:
        f.write(new_content)

if __name__ == "__main__":
    main()
