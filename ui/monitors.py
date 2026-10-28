#"""Мониторы: гистограмма светимости и профилометр среза яркости (Luminance Line Scan)."""
import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import QWidget, QDialog, QVBoxLayout, QLabel
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient, QFont, QPainterPath


from .widgets import AstroHistogramWidget, AstroHistogramHeader


class MiniHistogramWidget(AstroHistogramWidget):
    """Мини-гистограмма распределения яркости с предупреждением о пересветах кратеров."""
    pass



class ProfileMonitor(QDialog):
    """Окно отображения графика профиля яркости пикселей (0–255) вдоль проведенного отрезка."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Luminance Line Scan Monitor")
        self.resize(520, 220)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self._profile_data = np.zeros(100, dtype=np.float32)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.lbl_stats = QLabel("Distance: 0 px | Min: 0.0 | Max: 0.0 | Avg: 0.0")
        self.lbl_stats.setStyleSheet("color: #00fff2; font-size: 11px; font-weight: bold; background: transparent; border: none;")
        layout.addWidget(self.lbl_stats)

        self.canvas_widget = QWidget()
        self.canvas_widget.paintEvent = self._paint_graph
        layout.addWidget(self.canvas_widget)

        self.setStyleSheet("""
            QDialog {
                background-color: #141418;
                color: #dcdce6;
                border: 1px solid #2e2e3e;
                border-radius: 8px;
            }
            QLabel {
                background: transparent;
                border: none;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

    def update_profile(self, data: np.ndarray):
        if data is None or len(data) == 0:
            return
        self._profile_data = data
        min_v = float(np.min(data))
        max_v = float(np.max(data))
        avg_v = float(np.mean(data))
        self.lbl_stats.setText(f"Length: {len(data)} px | Min: {min_v*255:.1f} | Max: {max_v*255:.1f} | Avg: {avg_v*255:.1f}")
        self.canvas_widget.update()

    def _paint_graph(self, event):
        painter = QPainter(self.canvas_widget)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = float(self.canvas_widget.width())
        h = float(self.canvas_widget.height())

        clip_path = QPainterPath()
        clip_path.addRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 6.0, 6.0)
        painter.fillPath(clip_path, QBrush(QColor("#191921")))
        painter.strokePath(clip_path, QPen(QColor("#2b2b38"), 1.0))
        painter.setClipPath(clip_path)

        painter.setPen(QPen(QColor("#2b2b38"), 1, Qt.PenStyle.DashLine))
        for y_norm in [0.25, 0.50, 0.75]:
            y_pos = int(h * y_norm)
            painter.drawLine(0, y_pos, int(w), y_pos)

        n = len(self._profile_data)
        if n < 2:
            return

        path = QPainterPath()
        xs = np.linspace(0, w, n)
        ys = h - 1 - (self._profile_data * (h - 10))

        path.moveTo(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(x, y)

        fill_path = QPainterPath(path)
        fill_path.lineTo(w, h)
        fill_path.lineTo(0, h)
        fill_path.closeSubpath()

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(0, 173, 181, 160))
        grad.setColorAt(1.0, QColor(0, 173, 181, 15))
        painter.fillPath(fill_path, QBrush(grad))

        painter.setPen(QPen(QColor("#00fff2"), 2))
        painter.drawPath(path)
