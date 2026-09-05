import numpy as np

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class SpectrogramView(QWidget):
    """Compact, centered spectrogram for the current recording."""

    CORNER_RADIUS = 12

    # Size of the actual spectrogram.
    PLOT_WIDTH = 300
    PLOT_HEIGHT = 155

    # Space around the plot for axis labels/ticks. RIGHT_MARGIN also
    # holds the color legend (a colorbar showing what the colors mean).
    LEFT_MARGIN = 38
    RIGHT_MARGIN = 64
    TOP_MARGIN = 8
    BOTTOM_MARGIN = 36

    # Reduce the number of visual cells so the spectrogram
    # does not look like a collection of tiny vertical stripes.
    MAX_TIME_BINS = 180
    MAX_FREQ_BINS = 100

    def __init__(self, parent=None):
        super().__init__(parent)

        self._spectrum = None

        # Set via set_axis_params(). While they are None the axes fall
        # back to unitless 0.0 - 1.0 ticks.
        self._sample_rate = None
        self._hop_length = None

        self.setMinimumSize(
            self.PLOT_WIDTH
            + self.LEFT_MARGIN
            + self.RIGHT_MARGIN,
            self.PLOT_HEIGHT
            + self.TOP_MARGIN
            + self.BOTTOM_MARGIN
        )

        self.setMaximumHeight(205)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )

    def set_axis_params(self, sample_rate, hop_length):
        """
        Give the view the STFT parameters it needs to label the axes in
        Hz and seconds. Without them the ticks stay unitless.
        """
        self._sample_rate = sample_rate
        self._hop_length = hop_length
        self.update()

    def set_spectrum(self, spectrum):
        if spectrum is None:
            self._spectrum = None
        else:
            spectrum = np.asarray(spectrum)

            if spectrum.ndim == 2:
                # stft() returns (num_frames, n_fft). Keep the lower half
                # of the frequency axis (the signal is real, so the upper
                # half is a mirror copy) and transpose into the
                # (freq, time) layout the drawing code expects.
                n_freq = spectrum.shape[1] // 2 + 1
                spectrum = spectrum[:, :n_freq].T

            self._spectrum = spectrum

        self.update()

    def clear(self):
        self._spectrum = None
        self.update()

    @staticmethod
    def _downsample(data, max_freq, max_time):
        """Average bins for a cleaner visualization."""
        freq_bins, time_bins = data.shape

        target_time = min(time_bins, max_time)
        target_freq = min(freq_bins, max_freq)

        # Downsample time axis.
        if target_time < time_bins:
            edges = np.linspace(
                0,
                time_bins,
                target_time + 1,
                dtype=int,
            )

            time_data = np.empty(
                (freq_bins, target_time),
                dtype=float,
            )

            for i in range(target_time):
                start = edges[i]
                end = max(edges[i + 1], start + 1)

                time_data[:, i] = np.mean(
                    data[:, start:end],
                    axis=1,
                )
        else:
            time_data = data

        # Downsample frequency axis.
        current_freq = time_data.shape[0]

        if target_freq < current_freq:
            edges = np.linspace(
                0,
                current_freq,
                target_freq + 1,
                dtype=int,
            )

            result = np.empty(
                (target_freq, time_data.shape[1]),
                dtype=float,
            )

            for i in range(target_freq):
                start = edges[i]
                end = max(edges[i + 1], start + 1)

                result[i, :] = np.mean(
                    time_data[start:end, :],
                    axis=0,
                )

            return result

        return time_data

    @staticmethod
    def _color(value):
        """
        Dark navy -> purple -> magenta -> orange -> yellow.
        Strong spectral components become warmer/brighter.
        """
        value = float(np.clip(value, 0.0, 1.0))

        if value < 0.25:
            t = value / 0.25

            r = int(14 + 18 * t)
            g = int(16 + 12 * t)
            b = int(27 + 55 * t)

        elif value < 0.50:
            t = (value - 0.25) / 0.25

            r = int(32 + 48 * t)
            g = int(28 + 12 * t)
            b = int(82 + 55 * t)

        elif value < 0.75:
            t = (value - 0.50) / 0.25

            r = int(80 + 105 * t)
            g = int(40 + 20 * t)
            b = int(137 - 35 * t)

        else:
            t = (value - 0.75) / 0.25

            r = int(185 + 60 * t)
            g = int(60 + 150 * t)
            b = int(102 + 90 * t)

        return QColor(r, g, b)

    def _plot_geometry(self):
        """
        Center the plot together with the space needed for the
        frequency labels.
        """
        total_width = (
            self.PLOT_WIDTH
            + self.LEFT_MARGIN
            + self.RIGHT_MARGIN
        )

        group_left = max(
            0,
            (self.width() - total_width) // 2,
        )

        left = group_left + self.LEFT_MARGIN
        right = left + self.PLOT_WIDTH

        top = self.TOP_MARGIN
        bottom = top + self.PLOT_HEIGHT

        return (
            left,
            right,
            top,
            bottom,
            self.PLOT_WIDTH,
            self.PLOT_HEIGHT,
        )

    def _rounded_path(self, rect):
        path = QPainterPath()

        path.addRoundedRect(
            QRectF(rect),
            self.CORNER_RADIUS,
            self.CORNER_RADIUS,
        )

        return path

    def paintEvent(self, event):
        painter = QPainter(self)

        try:
            painter.setRenderHint(
                QPainter.RenderHint.Antialiasing
            )

            # -------------------------------------------------
            # Before recording
            # -------------------------------------------------

            if (
                self._spectrum is None
                or self._spectrum.size == 0
            ):
                painter.setPen(
                    QColor(125, 100, 120)
                )

                painter.setFont(
                    QFont("Sans Serif", 9)
                )

                painter.drawText(
                    self.rect(),
                    Qt.AlignmentFlag.AlignCenter,
                    "Waiting for recording...",
                )

                return

            # -------------------------------------------------
            # STFT
            # -------------------------------------------------

            spectrum = np.asarray(self._spectrum)

            if spectrum.ndim == 1:
                spectrum = spectrum[:, None]

            if spectrum.ndim != 2:
                painter.setPen(
                    QColor(150, 120, 145)
                )

                painter.drawText(
                    self.rect(),
                    Qt.AlignmentFlag.AlignCenter,
                    "Invalid spectrum",
                )

                return

            # Complex STFT -> magnitude.
            magnitude = np.abs(spectrum)

            if magnitude.size == 0:
                return

            # Magnitude -> dB.
            db = 20.0 * np.log10(
                np.maximum(magnitude, 1e-10)
            )

            low_db = float(
                np.percentile(db, 5)
            )

            high_db = float(
                np.percentile(db, 98)
            )

            if not np.isfinite(low_db):
                return

            if not np.isfinite(high_db):
                return

            if high_db <= low_db:
                high_db = low_db + 1.0

            normalized = np.clip(
                (db - low_db) / (high_db - low_db),
                0.0,
                1.0,
            )

            # Number of STFT frames before downsampling: the axis labels
            # describe the recording, not the reduced image.
            source_frames = spectrum.shape[1]

            # Reduce resolution for a cleaner image.
            normalized = self._downsample(
                normalized,
                self.MAX_FREQ_BINS,
                self.MAX_TIME_BINS,
            )

            freq_bins, time_bins = normalized.shape

            if freq_bins == 0 or time_bins == 0:
                return

            # -------------------------------------------------
            # Geometry
            # -------------------------------------------------

            (
                left,
                right,
                top,
                bottom,
                plot_width,
                plot_height,
            ) = self._plot_geometry()

            rect = QRectF(
                left,
                top,
                plot_width,
                plot_height,
            )

            path = self._rounded_path(rect)

            # -------------------------------------------------
            # Spectrogram
            # -------------------------------------------------

            painter.save()

            painter.setClipPath(path)

            painter.fillRect(
                rect,
                QColor(15, 14, 22),
            )

            for t in range(time_bins):
                x0 = int(
                    left
                    + t * plot_width / time_bins
                )

                x1 = max(
                    x0 + 1,
                    int(
                        left
                        + (t + 1)
                        * plot_width
                        / time_bins
                    ),
                )

                for f in range(freq_bins):
                    y0 = int(
                        top
                        + (
                            freq_bins
                            - 1
                            - f
                        )
                        * plot_height
                        / freq_bins
                    )

                    y1 = max(
                        y0 + 1,
                        int(
                            top
                            + (
                                freq_bins - f
                            )
                            * plot_height
                            / freq_bins
                        ),
                    )

                    painter.fillRect(
                        x0,
                        y0,
                        x1 - x0,
                        y1 - y0,
                        self._color(
                            normalized[f, t]
                        ),
                    )

            painter.restore()

            # -------------------------------------------------
            # Border
            # -------------------------------------------------

            painter.setPen(
                QPen(
                    QColor(105, 82, 105),
                    1,
                )
            )

            painter.drawPath(path)

            # -------------------------------------------------
            # Axis ticks
            # -------------------------------------------------

            painter.setFont(
                QFont("Sans Serif", 8)
            )

            painter.setPen(
                QColor(125, 110, 130)
            )

            # The top of the frequency axis is the Nyquist frequency,
            # because set_spectrum() keeps bins 0 .. n_fft//2.
            nyquist = (
                self._sample_rate / 2.0
                if self._sample_rate
                else None
            )

            # Recording length covered by the STFT frames.
            duration = (
                source_frames * self._hop_length / self._sample_rate
                if self._sample_rate and self._hop_length
                else None
            )

            # Frequency.
            for i in range(5):
                ratio = i / 4.0

                y = int(
                    bottom
                    - ratio * plot_height
                )

                painter.drawLine(
                    left - 3,
                    y,
                    left,
                    y,
                )

                if nyquist is None:
                    text = f"{ratio:.1f}"
                else:
                    text = f"{ratio * nyquist / 1000.0:.1f}"

                painter.drawText(
                    0,
                    y - 7,
                    left - 7,
                    14,
                    Qt.AlignmentFlag.AlignRight,
                    text,
                )

            # Time.
            for i in range(6):
                ratio = i / 5.0

                x = int(
                    left
                    + ratio * plot_width
                )

                painter.drawLine(
                    x,
                    bottom,
                    x,
                    bottom + 3,
                )

                if duration is None:
                    text = f"{ratio:.1f}"
                else:
                    text = f"{ratio * duration:.1f}"

                painter.drawText(
                    x - 15,
                    bottom + 5,
                    30,
                    14,
                    Qt.AlignmentFlag.AlignCenter,
                    text,
                )

            # -------------------------------------------------
            # Axis labels
            # -------------------------------------------------

            painter.setFont(
                QFont(
                    "Sans Serif",
                    8,
                    QFont.Weight.Bold,
                )
            )

            painter.setPen(
                QColor(110, 95, 115)
            )

            painter.drawText(
                0,
                top,
                left - 8,
                14,
                Qt.AlignmentFlag.AlignLeft,
                "kHz" if nyquist is not None else "Freq.",
            )

            painter.drawText(
                right - 32,
                bottom + 20,
                38,
                14,
                Qt.AlignmentFlag.AlignRight,
                "s" if duration is not None else "Time",
            )

            # -------------------------------------------------
            # Color legend (what the colors mean)
            # -------------------------------------------------

            bar_x0 = right + 10
            bar_width = 12
            bar_rect = QRectF(
                bar_x0, top, bar_width, plot_height
            )

            legend_gradient = QLinearGradient(
                bar_x0, top, bar_x0, bottom
            )

            for stop in (0.0, 0.25, 0.5, 0.75, 1.0):
                # Top of the bar = loudest (value 1.0),
                # bottom = quietest (value 0.0).
                legend_gradient.setColorAt(
                    stop, self._color(1.0 - stop)
                )

            painter.fillRect(bar_rect, legend_gradient)

            painter.setPen(
                QPen(QColor(105, 82, 105), 1)
            )
            painter.drawRect(bar_rect)

            painter.setFont(QFont("Sans Serif", 7))
            painter.setPen(QColor(115, 100, 120))

            label_x = int(bar_x0 + bar_width + 4)

            painter.drawText(
                label_x, top - 1, 34, 12,
                Qt.AlignmentFlag.AlignLeft,
                f"{high_db:.0f} dB",
            )

            painter.drawText(
                label_x, top + plot_height // 2 - 6, 34, 12,
                Qt.AlignmentFlag.AlignLeft,
                f"{(high_db + low_db) / 2:.0f}",
            )

            painter.drawText(
                label_x, bottom - 11, 34, 12,
                Qt.AlignmentFlag.AlignLeft,
                f"{low_db:.0f} dB",
            )

        finally:
            painter.end()


class AudioAnalysisWidget(QFrame):
    """Compact left-side panel containing only the spectrogram."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName(
            "audioAnalysisPanel"
        )

        self.setFrameShape(
            QFrame.Shape.NoFrame
        )

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            12,
            12,
            12,
            12,
        )

        layout.setSpacing(5)

        # -------------------------------
        # Title
        # -------------------------------

        title = QLabel(
            "Audio Analysis"
        )

        title.setObjectName(
            "audioAnalysisTitle"
        )

        layout.addWidget(title)

        # -------------------------------
        # Spectrogram (centered in the
        # remaining vertical space)
        # -------------------------------

        layout.addStretch(1)

        self.spectrogram = SpectrogramView()

        self.spectrogram.setObjectName(
            "spectrogramView"
        )

        layout.addWidget(
            self.spectrogram,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )

        layout.addStretch(1)

        # -------------------------------
        # Styling
        # -------------------------------

        self.setStyleSheet("""
            #audioAnalysisPanel {
                background-color: rgba(255, 255, 255, 28);
                border: none;
                border-radius: 16px;
            }

            #audioAnalysisTitle {
                color: #d85b99;
                font-size: 18px;
                font-weight: bold;
            }

            #spectrogramView {
                background: transparent;
                border: none;
            }
        """)

    def set_axis_params(self, sample_rate, hop_length):
        """Forward the STFT parameters used for the axis units."""
        self.spectrogram.set_axis_params(
            sample_rate,
            hop_length,
        )

    def set_audio_analysis(
        self,
        spectrum,
        peaks=None,
    ):
        """
        Display the current recording spectrum.

        `peaks` is intentionally ignored because this version
        contains only the spectrogram.
        """
        self.spectrogram.set_spectrum(
            spectrum
        )

    def clear(self):
        self.spectrogram.clear()