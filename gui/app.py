from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit
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

    def add_event(self, event):
        if "matched_company" in event.metadata:
            company = event.metadata["matched_company"]
            token = event.metadata.get("matched_token", "")
            event.content += f"\n\n🔎 Matched company: {company} (via token: '{token}')"
        self.text_log.append(f"[{event.timestamp}] {event.source}: {event.title}\n{event.content}\n")

def run_gui():
    app = QApplication(sys.argv)
    window = EventFeedGUI()
    window.show()
    return app, window