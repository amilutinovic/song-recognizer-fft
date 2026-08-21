"""
An animated "amoeba" button.
"""

import math

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QRadialGradient, QFont
from PyQt6.QtWidgets import QWidget


class AmoebaButton(QWidget):
    toggled_listening = pyqtSignal(bool)   # Emits True when it starts listening

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._spikiness = 0.0        # 0 = soft blob, 1 = sharp tentacles

        self._listening = False
        self._phase = 0.0            # Advances every frame to animate the edge
        self._amplitude = 0.06       # How far the edge wobbles (fraction of R)
        self._label = "listen"

        # Animation loop: ~60 fps.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    # Public API

    def set_listening(self, value: bool):
        """Turn the stronger 'listening' pulse on or off (no signal emitted)."""
        self._listening = value
        self._label = "listening" if value else "listen"

    def is_listening(self) -> bool:
        return self._listening

    # Interaction

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._listening = not self._listening
            self._label = "listening" if self._listening else "listen"
            self.toggled_listening.emit(self._listening)

    # Animation

    def _tick(self):
        # Idle = completely still. Listening = gentle, slightly spiky motion.
        if self._listening:
            speed = 0.045
            target_amp = 0.12
            target_spike = 1.0
        else:
            speed = 0.0            # frozen: no edge movement at rest
            target_amp = 0.05      # a small, static wobble so it's not a circle
            target_spike = 0.0
        self._phase += speed
        # Ease amplitude and spikiness toward their targets for smooth
        # transitions when listening starts/stops
        self._amplitude += (target_amp - self._amplitude) * 0.06
        self._spikiness += (target_spike - self._spikiness) * 0.05

        # Only repaint when something is actually changing: while listening,
        # or while a start/stop transition is still settling. At rest the
        # blob is frozen, so we skip repaints and don't waste CPU.
        settling = (abs(self._spikiness - target_spike) > 0.002 or
                    abs(self._amplitude - target_amp) > 0.002)
        if self._listening or settling:
            self.update()

    # Drawing

    def _blob_path(self, cx, cy, radius) -> QPainterPath:
        """Build the wobbly blob outline as a smooth closed path."""
        points = []
        n = 260                       # number of points around the edge
        for i in range(n):
            angle = 2 * math.pi * i / n
            # Low frequencies -> the soft, round base blob.
            base = (
                math.sin(3 * angle + self._phase) * 0.6
                + math.sin(5 * angle - self._phase * 1.3) * 0.3
                + math.sin(7 * angle + self._phase * 0.7) * 0.2
            )
            # High frequencies -> sharp tentacles. Faded in via spikiness,
            # so they only appear (and stick out) while listening.
            spikes = (
                math.sin(7 * angle - self._phase * 1.6) * 0.5
                + math.sin(11 * angle + self._phase * 2.1) * 0.35
                + math.sin(13 * angle - self._phase * 1.1) * 0.25
            )
            wobble = base + self._spikiness * spikes * 0.7
            r = radius * (1.0 + self._amplitude * wobble)
            points.append(QPointF(cx + r * math.cos(angle),
                                  cy + r * math.sin(angle)))

        # Build a smooth path through the points using quadratic segments
        # (midpoints as anchors) so the outline has no sharp corners.
        path = QPainterPath()
        mid_prev = (points[-1] + points[0]) / 2
        path.moveTo(mid_prev)
        for i in range(len(points)):
            p_curr = points[i]
            p_next = points[(i + 1) % len(points)]
            mid = (p_curr + p_next) / 2
            path.quadTo(p_curr, mid)
        path.closeSubpath()
        return path

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        radius = min(w, h) * 0.34

        # Soft outer glow: a few translucent blobs, growing and fading.
        for i, (scale, alpha) in enumerate([(1.28, 22), (1.16, 34), (1.06, 60)]):
            glow = self._blob_path(cx, cy, radius * scale)
            color = QColor(255, 105, 180, alpha)   # hot pink, translucent
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPath(glow)

        # Main blob with a radial gradient (lighter center -> hot pink edge).
        blob = self._blob_path(cx, cy, radius)
        gradient = QRadialGradient(cx, cy, radius)
        gradient.setColorAt(0.0, QColor(255, 150, 200))
        gradient.setColorAt(1.0, QColor(255, 90, 170))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(blob)

        # Label in the center.
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", int(radius * 0.28), QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._label)


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    button = AmoebaButton()
    button.setWindowTitle("Amoeba button preview")
    button.resize(500, 500)

    # Print state changes to the console for the preview.
    button.toggled_listening.connect(
        lambda on: print("listening" if on else "stopped"))

    button.show()
    sys.exit(app.exec())