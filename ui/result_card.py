"""
A single result card, Spotify-list style.

Shows one candidate song: a rank number, a colored cover tile with the
artist's initials (we generate it, since our indie tracks have no cover
art), the title and artist, and the match score / confidence.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


def _initials(text):
    """First letters of the first two words, e.g. 'Tom Orlando' -> 'TO'."""
    words = [w for w in text.replace("_", " ").split() if w]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


def _color_for(text):
    """A stable pink-ish color derived from the text, so the same song
    always gets the same cover tile."""
    h = sum(ord(c) for c in text)
    hues = [(255, 105, 180), (219, 112, 219), (255, 130, 170),
            (200, 120, 220), (255, 150, 160)]
    return QColor(*hues[h % len(hues)])


def _make_cover(text, size=48):
    """Draw a rounded square cover tile with the initials centered."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(_color_for(text))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(0, 0, size, size, 10, 10)
    painter.setPen(QColor(255, 255, 255))
    painter.setFont(QFont("Arial", int(size * 0.32), QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, _initials(text))
    painter.end()
    return pixmap


class ResultCard(QFrame):
    def __init__(self, rank, result, parent=None):
        super().__init__(parent)
        self.setObjectName("resultCard")
        self.setFixedHeight(66)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 12, 8)
        layout.setSpacing(12)

        # rank number
        rank_label = QLabel(str(rank))
        rank_label.setObjectName("rankLabel")
        rank_label.setFixedWidth(20)
        rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(rank_label)

        # generated cover tile
        cover = QLabel()
        cover.setPixmap(_make_cover(result["artist"] or result["title"]))
        layout.addWidget(cover)

        # title + artist stacked
        text_box = QVBoxLayout()
        text_box.setSpacing(2)
        title = QLabel(result["title"])
        title.setObjectName("titleLabel")
        artist = QLabel(result["artist"] or "Unknown artist")
        artist.setObjectName("artistLabel")
        text_box.addWidget(title)
        text_box.addWidget(artist)
        layout.addLayout(text_box, stretch=1)

        # score / confidence on the right
        score_box = QVBoxLayout()
        score_box.setSpacing(2)
        conf = QLabel(f"{result['confidence']:.0%}")
        conf.setObjectName("confLabel")
        conf.setAlignment(Qt.AlignmentFlag.AlignRight)
        score = QLabel(f"score {result['score']}")
        score.setObjectName("scoreLabel")
        score.setAlignment(Qt.AlignmentFlag.AlignRight)
        score_box.addWidget(conf)
        score_box.addWidget(score)
        layout.addLayout(score_box)