"""Стильный сплэш-скрин AstroRack DAW в тёмной эстетике DAW-платформы с логотипом."""
import os
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QSplashScreen, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QFrame, QApplication
)
from PyQt6.QtGui import QFont, QColor, QPalette, QPixmap, QIcon
from utils.file_io import get_resource_path


class AstroSplashScreen(QSplashScreen):
    """Сплэш-скрин AstroRack DAW v7.0 с анимацией загрузки, крупным логотипом и плавным затуханием (fade out)."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(540, 370)

        # Определение путей к ассетам
        icon_256_path = get_resource_path(os.path.join("assets", "icons", "icon_256x256.png"))
        icon_128_path = get_resource_path(os.path.join("assets", "icons", "icon_128x128.png"))
        icon_32_path = get_resource_path(os.path.join("assets", "icons", "icon_32x32.png"))
        ico_path = get_resource_path(os.path.join("assets", "app_icon.ico"))

        if os.path.exists(icon_256_path):
            self.setWindowIcon(QIcon(icon_256_path))
        elif os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))

        # Центрирование на экране
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            self.move((geo.width() - 540) // 2, (geo.height() - 370) // 2)

        self.setStyleSheet("""
            AstroSplashScreen {
                background-color: #0F1117;
                border: 1px solid #00adb5;
                border-radius: 8px;
            }
            QLabel {
                background: transparent;
            }
        """)

        # Главный контейнер
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 22, 28, 20)
        main_layout.setSpacing(0)

        # 1. Заголовок: Логотип и версия
        title_row = QHBoxLayout()
        title_row.setSpacing(10)

        lbl_top_icon = QLabel()
        lbl_top_icon.setFixedSize(28, 28)
        lbl_top_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.isfile(icon_32_path):
            lbl_top_icon.setPixmap(QPixmap(icon_32_path).scaled(26, 26, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            lbl_top_icon.setText("🔭")
            lbl_top_icon.setStyleSheet("font-size: 22px; color: #00fff2;")
        title_row.addWidget(lbl_top_icon)

        lbl_title = QLabel("AstroRack DAW")
        lbl_title.setStyleSheet("""
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: 0.5px;
            color: #f1f5f9;
        """)
        title_row.addWidget(lbl_title)

        lbl_badge = QLabel("v7.0")
        lbl_badge.setStyleSheet("""
            background-color: #14383c;
            color: #00fff2;
            border: 1px solid #00adb5;
            border-radius: 4px;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11px;
            font-weight: bold;
            padding: 3px 8px;
            margin-bottom: 2px;
        """)
        title_row.addWidget(lbl_badge)
        title_row.addStretch()

        main_layout.addLayout(title_row)
        main_layout.addSpacing(4)

        # 2. Подзаголовок
        lbl_subtitle = QLabel("Lunar, Planetary & Deep-Sky Image Mastering Platform")
        lbl_subtitle.setStyleSheet("""
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11px;
            font-weight: 500;
            color: #94a3b8;
            margin-left: 2px;
        """)
        main_layout.addWidget(lbl_subtitle)
        main_layout.addSpacing(10)

        # 3. Тонкий разделитель
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Plain)
        sep.setStyleSheet("background-color: #1e2433; color: #1e2433; max-height: 1px;")
        main_layout.addWidget(sep)
        main_layout.addSpacing(8)

        # 4. Манифест
        lbl_manifest = QLabel("OPEN SOURCE • NO RIGHTS RESERVED • BUILT FOR STARGAZERS")
        lbl_manifest.setStyleSheet("""
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 1.2px;
            color: #475569;
            margin-left: 2px;
        """)
        main_layout.addWidget(lbl_manifest)

        main_layout.addSpacing(8)

        # 5. Крупный центральный логотип 128x128 px
        center_logo_row = QHBoxLayout()
        center_logo_row.addStretch()
        self.lbl_center_logo = QLabel()
        self.lbl_center_logo.setFixedSize(128, 128)
        self.lbl_center_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.isfile(icon_128_path):
            self.lbl_center_logo.setPixmap(QPixmap(icon_128_path))
        center_logo_row.addWidget(self.lbl_center_logo)
        center_logo_row.addStretch()
        main_layout.addLayout(center_logo_row)

        main_layout.addSpacing(10)

        # 6. Статус инициализации и прогресс-бар
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("Initializing DSP core...")
        self.lbl_status.setStyleSheet("""
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11px;
            font-weight: 600;
            color: #00adb5;
        """)
        status_row.addWidget(self.lbl_status)
        status_row.addStretch()

        self.lbl_percent = QLabel("0%")
        self.lbl_percent.setStyleSheet("""
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 10px;
            font-weight: bold;
            color: #64748b;
        """)
        status_row.addWidget(self.lbl_percent)
        main_layout.addLayout(status_row)
        main_layout.addSpacing(6)

        # Тонкий прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1e2b;
                border: none;
                border-radius: 1px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00adb5, stop:1 #00fff2);
                border-radius: 1px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        self._fade_anim = None

    def set_progress(self, percent: int, message: str):
        """Обновление шкалы прогресса и лога этапа инициализации."""
        self.progress_bar.setValue(int(percent))
        self.lbl_status.setText(str(message))
        self.lbl_percent.setText(f"{int(percent)}%")
        QApplication.processEvents()

    def fade_out_and_finish(self, main_window, duration_ms: int = 350):
        """Плавное затухание сплэша (fade out) и последующий вывод главного окна."""
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(duration_ms)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        def _on_finished():
            self.close()
            main_window.showMaximized()

        self._fade_anim.finished.connect(_on_finished)
        self._fade_anim.start()
