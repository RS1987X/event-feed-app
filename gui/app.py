from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtCore import pyqtSlot
import sys

class EventFeedGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Event Feed")
        self.resize(600, 400)
        self.layout = QVBoxLayout()
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.layout.addWidget(self.text_log)
        self.setLayout(self.layout)

    # @pyqtSlot(object)
    # def add_event(self, event):
    #     if "matched_company" in event.metadata:
    #         company = event.metadata["matched_company"]
    #         token = event.metadata.get("matched_token", "")
    #         event.content += f"\n\n🔎 Matched company: {company} (via token: '{token}')"
    #     self.text_log.append(f"[{event.timestamp}] {event.source}: {event.title}\n{event.content}\n")
    
    @pyqtSlot(object)
    def add_event(self, event):
        # 1) If we bundled all matches into one Event…
        if "matches" in event.metadata:
            for company, token in event.metadata["matches"]:
                event.content += f"\n🔎 {company} (via token: '{token}')"

        # 2) …otherwise, fall back to the old single‐match fields
        elif "matched_company" in event.metadata:
            company = event.metadata["matched_company"]
            token   = event.metadata.get("matched_token", "")
            event.content += f"\n🔎 Matched company: {company} (via token: '{token}')"

        # 3) Finally append to the log
        self.text_log.append(
            f"[{event.timestamp}] {event.source}: {event.title}\n{event.content}\n"
        )
def run_gui():
    app = QApplication(sys.argv)
    window = EventFeedGUI()
    window.show()
    return app, window