import sys

def main():
    with open("Analysis/AnalysisOfflineGUI.py", "r") as f:
        content = f.read()

    start_str = "    def start_bsrh_analysis(self):"
    end_str = "        self.worker.start()"

    start_idx = content.find(start_str)
    end_idx = content.find(end_str, start_idx) + len(end_str)

    rep = """    def start_bsrh_analysis(self):
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
        self.worker.start()"""

    new_content = content[:start_idx] + rep + content[end_idx:]

    with open("Analysis/AnalysisOfflineGUI.py", "w") as f:
        f.write(new_content)

if __name__ == "__main__":
    main()
