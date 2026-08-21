"""
Main application window.

Flow:
  click amoeba -> start MicRecorder, blob starts pulsing
  click again  -> stop recording, hand the signal to RecognizerWorker
  worker done  -> show results (or "not recognized") on the right
"""

import os
import sys

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QVBoxLayout,
                             QWidget, QFrame)

# Make sibling packages importable when running this file directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "audio"))
for sub in ("ui", "audio", "fourier", "fingerprints"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from amoeba_button import AmoebaButton
from recognizer_worker import RecognizerWorker
from result_card import ResultCard
from audio_io import MicRecorder

DB_PATH = os.path.join(ROOT, "data", "fingerprints.db")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("listen")
        self.resize(1100, 620)

        self._recorder = MicRecorder()
        self._worker = None

        self._build_ui()
        self._apply_style()

    # --- layout ----------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # LEFT: reserved for the colleague's live spectrogram. Left empty
        # so she can decide how to fill it (inline view or a popup).
        self.spectrogram_panel = QFrame()
        self.spectrogram_panel.setObjectName("spectrogramPanel")
        root.addWidget(self.spectrogram_panel, stretch=5)

        # MIDDLE: amoeba button + status line.
        center = QVBoxLayout()
        center.setContentsMargins(20, 20, 20, 20)
        center.addStretch(1)
        self.amoeba = AmoebaButton()
        self.amoeba.toggled_listening.connect(self._on_toggle)
        center.addWidget(self.amoeba, alignment=Qt.AlignmentFlag.AlignCenter)
        self.status_label = QLabel("Tap to listen")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.status_label)
        center.addStretch(1)
        center_wrap = QWidget()
        center_wrap.setLayout(center)
        root.addWidget(center_wrap, stretch=4)

        # RIGHT: results panel.
        self.results_panel = QFrame()
        self.results_panel.setObjectName("resultsPanel")
        self.results_layout = QVBoxLayout(self.results_panel)
        self.results_layout.setContentsMargins(16, 20, 16, 20)
        self.results_layout.setSpacing(10)
        self.results_title = QLabel("Results")
        self.results_title.setObjectName("resultsTitle")
        self.results_layout.addWidget(self.results_title)
        self.results_layout.addStretch(1)
        root.addWidget(self.results_panel, stretch=5)

    # --- interaction -----------------------------------------------------

    def _on_toggle(self, listening):
        if listening:
            self._start_listening()
        else:
            self._stop_and_recognize()

    def _start_listening(self):
        self._clear_results()
        self.status_label.setText("Listening... tap again to stop")
        try:
            self._recorder.start()
        except Exception as exc:
            self.status_label.setText(f"Microphone error: {exc}")
            self.amoeba.set_listening(False)

    def _stop_and_recognize(self):
        self.status_label.setText("Analyzing...")
        signal = self._recorder.stop()

        self._worker = RecognizerWorker(DB_PATH, signal=signal)
        self._worker.status.connect(self.status_label.setText)
        self._worker.finished.connect(self._on_results)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    # --- results ---------------------------------------------------------

    def _on_results(self, recognized, results):
        self._clear_results()
        if recognized and results:
            self.status_label.setText("Tap to listen")
            self.results_title.setText("Top matches")
            for i, result in enumerate(results, 1):
                self.results_layout.insertWidget(
                    self.results_layout.count() - 1, ResultCard(i, result))
        else:
            self.status_label.setText("Tap to listen")
            self.results_title.setText("No match")
            msg = QLabel("No song in the database matched this audio.")
            msg.setObjectName("noMatchLabel")
            msg.setWordWrap(True)
            self.results_layout.insertWidget(
                self.results_layout.count() - 1, msg)
            
    def _on_failed(self, message):
        self.amoeba.set_listening(False)
        self.status_label.setText("Tap to listen")
        self._clear_results()
        self.results_title.setText("Error")
        msg = QLabel(message)
        msg.setObjectName("noMatchLabel")
        msg.setWordWrap(True)
        self.results_layout.insertWidget(
            self.results_layout.count() - 1, msg)

    def _clear_results(self):
        # Remove every widget except the title (index 0) and the stretch.
        while self.results_layout.count() > 2:
            item = self.results_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

    # --- background ------------------------------------------------------

    def paintEvent(self, event):
        """Paint the soft pastel diagonal gradient from the design:
        pink (top-left) -> lavender -> peach -> pale yellow (bottom-right)."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(QPointF(0, 0),
                                   QPointF(self.width(), self.height()))
        gradient.setColorAt(0.0, QColor(236, 160, 224))   # stronger pink
        gradient.setColorAt(0.30, QColor(245, 180, 216))  # pink-lavender
        gradient.setColorAt(0.58, QColor(252, 206, 190))  # warm peach
        gradient.setColorAt(1.0, QColor(250, 236, 168))   # pale yellow
        painter.fillRect(self.rect(), gradient)

    # --- style -----------------------------------------------------------

    def _apply_style(self):
        # Panels are transparent so the painted gradient shows through.
        # Text and cards use dark/pink tones that read well on the light
        # background. The results panel gets a soft translucent white so the
        # cards stand out, echoing the framed box in the design.
        self.setStyleSheet("""
            QWidget { background-color: transparent; color: #4a2f3f; }
            #spectrogramPanel { background-color: transparent; }
            #statusLabel { color: #a3436f; font-size: 15px; font-weight: bold; }
            #resultsPanel {
                background-color: rgba(255, 255, 255, 45);
                border-radius: 18px;
            }
            #resultsTitle { color: #e0559a; font-size: 20px; font-weight: bold; }
            #resultCard {
                background-color: rgba(255, 255, 255, 245);
                border-radius: 10px;
            }
            #rankLabel { color: #e0559a; font-size: 18px; font-weight: bold; }
            #titleLabel { color: #1a1418; font-size: 15px; font-weight: bold; }
            #artistLabel { color: #6a5560; font-size: 12px; }
            #confLabel { color: #e0559a; font-size: 14px; font-weight: bold; }
            #scoreLabel { color: #9a8490; font-size: 11px; }
            #noMatchLabel { color: #7a5266; font-size: 14px; font-weight: bold; }
        """)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()