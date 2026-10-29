#"""Специализированные виджеты: PlanetarySlider, CollapsibleInsertBox, HelpIcon, AstroParamRow."""
import datetime
from typing import Optional, Union, Callable
import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRectF, QTimer, QObject, QEvent, QDate, QSize
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QSlider, QCheckBox, QPushButton, QDialog,
    QToolButton, QDoubleSpinBox, QDateEdit, QRadioButton, QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox,
    QMenu, QToolTip,
    QStyledItemDelegate,
    QSizePolicy, QButtonGroup
)
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPainterPath, QPalette, QFont, QCursor, QPolygonF,
    QLinearGradient, QRadialGradient
)
from core.processing import AstroEngine
from core.lunar_distance import calculate_lunar_distance, DEFAULT_LUNAR_DISTANCE_KM

THEMES = {
    'A': {
        'primary': '#00adb5',
        'bright': '#00fff2',
        'bg_accent': '#14383c',
        'bg_hover': '#1c4e54',
        'border_hover': '#00adb5',
        'name': 'A',
    },
    'B': {
        'primary': '#f59e0b',
        'bright': '#fbbf24',
        'bg_accent': '#382810',
        'bg_hover': '#4e3814',
        'border_hover': '#f59e0b',
        'name': 'B',
    }
}

ASTRO_HELP_TEXTS = {
    "Sharpness Mode": (
        "<b>Выбор математического алгоритма повышения резкости</b><br><br>"
        "• <b>Wavelet (à trous)</b>: многомасштабное разложение сигнала по B3-сплайнам. "
        "Позволяет раздельно усиливать тончайшие структуры (кратеры, полосы) без раздувания средних масштабов.<br><br>"
        "• <b>Unsharp Mask (Нерезкое маскирование)</b>: классический фото- и астрономический алгоритм вычитания "
        "размытой копии (Gaussian blur) для акцентирования перепадов яркости."
    ),
    "Amount (Strength)": (
        "<b>Сила нерезкого маскирования (Amount)</b><br><br>"
        "Коэффициент усиления высокочастотного перепада яркости (до 15.0x). "
        "Увеличивает локальный контраст границ деталей."
    ),
    "Radius (Sigma Blur)": (
        "<b>Радиус размытия маски (Sigma)</b><br><br>"
        "Определяет пространственный масштаб деталей, затрагиваемых маской нерезкости (до 20.0 px). "
        "Малые значения (1.0–2.0) акцентируют микродетали, большие — общий рельеф."
    ),
    "Threshold (Noise Floor)": (
        "<b>Порог отсечки шума (Threshold)</b><br><br>"
        "Минимальная разница яркости между пикселями, к которой применяется усиление. "
        "Предотвращает усиление гранулированного шума матрицы на гладких участках (лунные моря, диск планеты)."
    ),
    "Layer 1 (Fine Details)": (
        "<b>Слой 1 à trous вейвлет-декомпозиции (1–2 px)</b><br><br>"
        "Выделяет тончайшие высокочастотные (HF) структуры: мелкие лунные кратеры, "
        "гребни валов, фестоны Юпитера и деления колец Сатурна.<br><br>"
        "Диапазон расширен до 10.00 для экстремальной проработки микрорельефа."
    ),
    "Layer 2 (Medium Details)": (
        "<b>Слой 2 à trous вейвлет-декомпозиции (2–4 px)</b><br><br>"
        "Усиливает структуры среднего масштаба (MF): террасы валов кратеров, центральные горки, "
        "лучевые системы и контраст облачных поясов планет."
    ),
    "Denoise (Layer 1 Threshold)": (
        "<b>Пороговое шумоподавление (Threshold Denoise) 1-го слоя</b><br><br>"
        "Срезает высокочастотный шум матрицы ДО усиления деталей."
    ),
    "Radius (Sigma PSF)": (
        "<b>Радиус функции рассеяния точки (PSF) в пикселях</b><br><br>"
        "Математическая модель оптического и атмосферного размытия (Seeing)."
    ),
    "Iterations (Deconvolution)": (
        "<b>Итерации деконволюции Ричардсона-Люси</b><br><br>"
        "Количество циклов восстановления истинного резкого сигнала."
    ),
    "Protect Edge (Anti-Ringing)": (
        "<b>Защита лимба от артефактов Гиббса</b><br><br>"
        "Формирует маску краевого перепада «планета-космос» и ослабляет деконволюцию на границе, "
        "устраняя жесткий белый ореол вокруг планеты."
    ),
    "Black Level (Sky Darkening)": (
        "<b>Отсечка точки черного</b><br><br>"
        "Вычитает рассеянный свет в оптике и засветку неба, делая космический фон черным."
    ),
    "Midtones / Gamma": (
        "<b>Гамма средних тонов</b><br><br>"
        "Проявляет детали в лунных морях, не пережигая светлые плато и террасы."
    ),
    "Contrast (S-Curve)": (
        "<b>S-образная кривая контраста</b><br><br>"
        "Усиливает микроконтраст на границе света и тени вдоль линии терминатора."
    ),
    "Extract Green Channel": (
        "<b>Извлечение зеленого канала (RGGB)</b><br><br>"
        "Зеленый канал обладает наивысшим пространственным разрешением и наилучшим SNR."
    ),
    "Auto RGB Align": (
        "<b>Авто-выравнивание каналов (Atmospheric Dispersion Fix)</b><br><br>"
        "Субпиксельное совмещение R и B каналов относительно G через фазовую корреляцию "
        "для устранения цветных ободков атмосферной дисперсии."
    ),
    "Local Contrast Mode": (
        "<b>Выбор математического алгоритма локального контраста (Clarity)</b><br><br>"
        "• <b>CLAHE (Contrast Limited Adaptive Histogram Equalization)</b>: адаптивное выравнивание локальных гистограмм "
        "по сетке тайлов с ограничением контраста. Усиливает видимость малоконтрастных лунных морей, кратеров и микрорельефа без выгорания светов.<br><br>"
        "• <b>High-Pass (Макро-нерезкость)</b>: широкоапертурное усиление через вычитание крупномасштабного размытия по Гауссу "
        "для разделения крупных структур и придания объемности рельефу."
    ),
    "Clip Limit (Contrast)": (
        "<b>Порог ограничения контраста CLAHE (Clip Limit)</b><br><br>"
        "Задает максимальный коэффициент усиления локального контраста в пределах каждого тайла (1.00–5.00x). "
        "При значении 1.00 усиление нейтрально. Предохраняет света и тени от жесткого клиппинга."
    ),
    "Grid Size (Tile Area)": (
        "<b>Размер сетки тайлов CLAHE (Tile Grid Size)</b><br><br>"
        "Определяет размер локальной окрестности анализа гистограммы (от 8×8 до 64×64 px). "
        "Меньший размер усиливает тонкие микроструктуры, больший размер дает плавные макропереходы."
    ),
    "Clarity Amount": (
        "<b>Сила макроконтраста High-Pass</b><br><br>"
        "Коэффициент подмешивания высокочастотной макроразности к изображению (0.00–3.00x). "
        "Позволяет подчеркнуть перепады освещенности между светлыми плато и темными бассейнами."
    ),
    "Sigma Radius (Blur)": (
        "<b>Радиус размытия High-Pass (Sigma Blur)</b><br><br>"
        "Радиус функции Гаусса (15–60 px), задающий пространственный масштаб объемных макроструктур."
    ),
    "Color Saturation": (
        "<b>Насыщенность цвета (Color Saturation & Vibrance)</b><br><br>"
        "Масштабирование каналов цветности без изменения яркости Rec.601 (Luma). "
        "Предотвращает пересвет и появление цветового шума в ярких участках кадра."
    ),
    "Mineral Moon Boost (Ti / Fe)": (
        "<b>Минеральное насыщение Луны (Mineral Moon Boost)</b><br><br>"
        "Избирательное нелинейное насыщение синих оттенков (богатый титаном реголит в Море Спокойствия) "
        "и оранжево-красных оттенков (оксиды железа в морских бассейнах) с подавлением паразитного зеленого шума матрицы."
    ),
    "Neutralize Sky Background": (
        "<b>Нейтрализация космического фона</b><br><br>"
        "Вычисляет смещение цветовых каналов (R, G, B) на темных участках неба "
        "и приводит космический фон к нейтральному темно-серому/черному уровню без среза теней терминатора."
    ),
    "Dry / Wet Mix (%)": (
        "<b>Параллельное подмешивание исходного сигнала (Dry/Wet)</b><br><br>"
        "Определяет баланс между входным сигналом инсерта (Dry, 0%) и обработанным результатом (Wet, 100%). "
        "Позволяет осуществлять параллельную обработку (параллельное повышение резкости, мягкую деконволюцию или локальный контраст)."
    )
}



class PresetComboDelegate(QStyledItemDelegate):
    """Кастомный делегат для стилизации пунктов сохранения/загрузки в янтарно-золотой цвет."""
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.data(Qt.ItemDataRole.UserRole + 1) is True:
            option.palette.setColor(QPalette.ColorRole.Text, QColor("#e5a93b"))
            option.palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffe8a0"))
            option.font.setBold(True)


class AstroHelpDialog(QDialog):
    """Стилизованное модальное окно справки с переносом строк и фиксированной шириной."""

    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Справка: {title}")
        self.setFixedWidth(420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.lbl_text = QLabel()
        self.lbl_text.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_text.setText(f"<div style='font-size: 11px; line-height: 155%; color: #dcdce6;'>{text}</div>")
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setStyleSheet("background: transparent; border: none; padding: 0;")
        layout.addWidget(self.lbl_text)

        btn_box = QHBoxLayout()
        btn_box.addStretch()
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.accept)
        btn_box.addWidget(self.btn_close)
        layout.addLayout(btn_box)

        self.setStyleSheet("""
            QDialog {
                background-color: #16161e;
                border: 1px solid #2e2e3e;
                border-radius: 8px;
            }
            QLabel {
                background: transparent;
                border: none;
                color: #dcdce6;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #22222e;
                color: #00fff2;
                border: 1px solid #00adb5;
                border-radius: 5px;
                padding: 5px 22px;
                font-weight: bold;
                font-size: 11px;
                min-width: 70px;
            }
            QPushButton:hover {
                background-color: #00adb5;
                color: #111114;
                border-color: #00fff2;
            }
            QPushButton:pressed {
                background-color: #00888f;
                color: #ffffff;
            }
        """)


def show_astro_help(parent, title: str):
    """Показывает информационное модальное окно с описанием физического смысла параметра."""
    text = ASTRO_HELP_TEXTS.get(title, "Описание параметра отсутствует.")
    dlg = AstroHelpDialog(title, text, parent)
    dlg.exec()


class HelpIcon(QToolButton):
    """Кнопка '?' для вызова справки."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.param_title = title
        self.setText("?")
        self.setFixedSize(14, 14)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Физический смысл параметра")
        self._theme = THEMES['A']
        self._apply_style()
        self.clicked.connect(lambda: show_astro_help(self, self.param_title))

    def _apply_style(self):
        primary = self._theme.get('primary', '#00adb5') if hasattr(self, '_theme') and self._theme else '#00adb5'
        bright = self._theme.get('bright', '#00fff2') if hasattr(self, '_theme') and self._theme else '#00fff2'
        self.setStyleSheet(f"""
            QToolButton {{
                background-color: #242432;
                color: {bright};
                font-weight: bold;
                font-size: 9px;
                border-radius: 7px;
                border: 1px solid {primary};
                padding: 0;
            }}
            QToolButton:hover {{
                background-color: {primary};
                color: #111114;
            }}
            QToolButton:disabled {{
                background-color: #161922;
                color: #475569;
                border: 1px solid #334155;
            }}
        """)

    def set_theme(self, theme: dict):
        self._theme = theme
        self._apply_style()


class PlanetarySlider(QSlider):
    """
    Кастомный слайдер Солнечной системы:
    Размер ручки: от 8 px (Меркурий) до 22 px (Юпитер).
    0% - 15%: Меркурий/Луна (#a0a0a0)
    15% - 35%: Марс (#e76f51)
    35% - 60%: Земля/Нептун (#00adb5)
    60% - 85%: Сатурн (#e9c46a с кольцом)
    85% - 100%: Юпитер (#f4a261 с полосами)
    """
    doubleClicked = pyqtSignal()

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setRange(0, 10000)
        self.setMouseTracking(True)
        self.setFixedHeight(22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme = THEMES['A']

    def set_theme(self, theme: dict):
        self._theme = theme
        self.update()

    def mouseDoubleClickEvent(self, event):
        if not self.isEnabled():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if not self.isEnabled():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._update_from_mouse(event.pos().x())
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.isEnabled():
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._update_from_mouse(event.pos().x())
            event.accept()
        super().mouseMoveEvent(event)

    def _update_from_mouse(self, mouse_x):
        margin_x = 12.0
        w = max(1.0, float(self.width()) - 2.0 * margin_x)
        t = np.clip((float(mouse_x) - margin_x) / w, 0.0, 1.0)
        val = int(round(self.minimum() + t * (self.maximum() - self.minimum())))
        self.setValue(val)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = float(self.width())
        h = float(self.height())
        margin_x = 12.0

        val_range = max(1, self.maximum() - self.minimum())
        t = float(self.value() - self.minimum()) / float(val_range)
        t = np.clip(t, 0.0, 1.0)

        track_y = h / 2.0 - 2.0
        track_h = 4.0
        track_w = max(1.0, w - 2.0 * margin_x)
        track_rect = QRectF(margin_x, track_y, track_w, track_h)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#242430")))
        painter.drawRoundedRect(track_rect, 2.0, 2.0)

        diam = 8.0 + 14.0 * t
        r = diam / 2.0

        handle_x = margin_x + t * track_w
        handle_y = h / 2.0

        active_w = handle_x - margin_x
        if active_w > 0:
            active_rect = QRectF(margin_x, track_y, active_w, track_h)
            if not self.isEnabled():
                fill_col = QColor("#2b2b36")
            elif self._theme is not None and 'primary' in self._theme:
                fill_col = QColor(self._theme['primary'])
            elif t < 0.15:
                fill_col = QColor("#808080")
            elif t < 0.35:
                fill_col = QColor("#e76f51")
            elif t < 0.60:
                fill_col = QColor("#00adb5")
            elif t < 0.85:
                fill_col = QColor("#e9c46a")
            else:
                fill_col = QColor("#f4a261")
            painter.setBrush(QBrush(fill_col))
            painter.drawRoundedRect(active_rect, 2.0, 2.0)

        diam = 8.0 + 14.0 * t
        r = diam / 2.0

        if not self.isEnabled():
            painter.setPen(QPen(QColor("#3a3a46"), 1))
            painter.setBrush(QBrush(QColor("#202028")))
            painter.drawEllipse(QPointF(handle_x, handle_y), r, r)
            return

        if t < 0.15:
            painter.setPen(QPen(QColor("#dcdcdc"), 1.2))
            painter.setBrush(QBrush(QColor("#a0a0a0")))
            painter.drawEllipse(QPointF(handle_x, handle_y), r, r)
        elif t < 0.35:
            painter.setPen(QPen(QColor("#ff8c72"), 1.2))
            painter.setBrush(QBrush(QColor("#e76f51")))
            painter.drawEllipse(QPointF(handle_x, handle_y), r, r)
        elif t < 0.60:
            painter.setPen(QPen(QColor("#00fff2"), 1.2))
            painter.setBrush(QBrush(QColor("#00adb5")))
            painter.drawEllipse(QPointF(handle_x, handle_y), r, r)
        elif t < 0.85:
            ring_pen = QPen(QColor("#f4e285"), 1.6)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(handle_x, handle_y), r * 1.55, r * 0.55)

            painter.setPen(QPen(QColor("#fff2a8"), 1.2))
            painter.setBrush(QBrush(QColor("#e9c46a")))
            painter.drawEllipse(QPointF(handle_x, handle_y), r, r)
        else:
            painter.setPen(QPen(QColor("#ffe0be"), 1.4))
            painter.setBrush(QBrush(QColor("#f4a261")))
            painter.drawEllipse(QPointF(handle_x, handle_y), r, r)

            painter.save()
            clip_path = QPainterPath()
            clip_path.addEllipse(QPointF(handle_x, handle_y), r, r)
            painter.setClipPath(clip_path)

            belt_pen = QPen(QColor("#c46b38"), 1.8)
            painter.setPen(belt_pen)
            painter.drawLine(int(handle_x - r), int(handle_y - 2.5), int(handle_x + r), int(handle_y - 2.5))
            painter.drawLine(int(handle_x - r), int(handle_y + 2.5), int(handle_x + r), int(handle_y + 2.5))
            painter.restore()


class AstroParamRow(QWidget):
    """Строка параметра с суб-байпасом (опционально), ручкой-планетой и точным спинбоксом."""
    valueChanged = pyqtSignal(float)
    bypassToggled = pyqtSignal()
    paramCommitted = pyqtSignal()

    def __init__(self, title: str, min_val: float, max_val: float,
                 default_val: float, step: float, decimals: int = 2,
                 neutral_val: float = 1.0, has_sub_bypass: bool = True, parent=None):
        super().__init__(parent)
        self.param_title = title
        self.min_val = min_val
        self.max_val = max_val
        self.default_val = default_val
        self.step = step
        self.decimals = decimals
        self.neutral_val = neutral_val
        self.has_sub_bypass = has_sub_bypass
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(0, 0, 0, 0)
        main_vbox.setSpacing(4)

        header_hbox = QHBoxLayout()
        header_hbox.setContentsMargins(0, 0, 0, 0)
        header_hbox.setSpacing(6)
        header_hbox.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        if self.has_sub_bypass:
            self.chk_sub_bypass = QCheckBox()
            self.chk_sub_bypass.setChecked(False)
            self.chk_sub_bypass.setFixedSize(14, 14)
            self.chk_sub_bypass.setToolTip("Включить / Обойти данный параметр (Sub-Bypass)")
            self.chk_sub_bypass.setStyleSheet("""
                QCheckBox {
                    margin: 0px;
                    padding: 0px;
                }
                QCheckBox::indicator {
                    width: 12px;
                    height: 12px;
                    margin: 0px;
                }
                QCheckBox::indicator:checked {
                    background-color: #00adb5;
                    border: 1px solid #00fff2;
                    border-radius: 3px;
                }
                QCheckBox::indicator:unchecked {
                    background-color: #23232c;
                    border: 1px solid #454555;
                    border-radius: 3px;
                }
                QCheckBox::indicator:checked:disabled {
                    background-color: #334155;
                    border: 1px solid #475569;
                    border-radius: 3px;
                    image: url("assets/icons/check_ghost.png");
                }
                QCheckBox::indicator:unchecked:disabled {
                    background-color: #1a1a22;
                    border: 1px solid #334155;
                    border-radius: 3px;
                }
            """)
            self.chk_sub_bypass.toggled.connect(self._on_sub_bypass_changed)
            header_hbox.addWidget(self.chk_sub_bypass)
        else:
            self.chk_sub_bypass = None

        self.lbl_title = ElidedLabel(title, self)
        self.lbl_title.setStyleSheet("color: #777788; font-size: 10px; font-weight: bold;" if self.has_sub_bypass else "color: #b0b0be; font-size: 10px; font-weight: bold;")

        self.btn_help = HelpIcon(self.param_title)

        self.btn_reset = QPushButton("R")
        self.btn_reset.setFixedSize(14, 14)
        self.btn_reset.setToolTip("Сбросить значение к умолчанию")
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #242432;
                color: #ff9900;
                font-weight: bold;
                font-size: 8px;
                border-radius: 3px;
                border: 1px solid #554422;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #ff9900;
                color: #111114;
            }
            QPushButton:disabled {
                background-color: #161922;
                color: #475569;
                border: 1px solid #334155;
            }
        """)
        self.btn_reset.clicked.connect(self.reset_to_default)

        header_hbox.addWidget(self.lbl_title)
        header_hbox.addWidget(self.btn_help)
        header_hbox.addWidget(self.btn_reset)
        header_hbox.addStretch()
        main_vbox.addLayout(header_hbox)

        ctrl_hbox = QHBoxLayout()
        ctrl_hbox.setContentsMargins(0, 0, 0, 0)
        ctrl_hbox.setSpacing(8)

        self.slider = PlanetarySlider(Qt.Orientation.Horizontal)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.setMinimumWidth(50)
        self.slider.setValue(self._val_to_slider(default_val))
        self.slider.doubleClicked.connect(self.reset_to_default)
        self.slider.setEnabled(not self.has_sub_bypass)
        self.slider.sliderReleased.connect(self.paramCommitted.emit)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setValue(default_val)
        self.spinbox.setFixedWidth(44)
        self.spinbox.setFixedHeight(20)
        self.spinbox.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spinbox.setEnabled(not self.has_sub_bypass)
        self.spinbox.editingFinished.connect(self.paramCommitted.emit)
        self.spinbox.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #1a1a22;
                border: 1px solid #363644;
                border-radius: 4px;
                color: #00fff2;
                font-size: 10px;
                padding: 0 2px;
            }
            QDoubleSpinBox:disabled {
                color: #555562;
                border-color: #262630;
            }
        """)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

        ctrl_hbox.addWidget(self.slider, 1)
        ctrl_hbox.addWidget(self.spinbox, 0)
        main_vbox.addLayout(ctrl_hbox)

    def apply_theme(self, theme: dict):
        self._theme = theme
        primary = theme['primary']
        bright = theme['bright']
        if self.has_sub_bypass and self.chk_sub_bypass is not None:
            self.chk_sub_bypass.setStyleSheet(f"""
                QCheckBox {{
                    background: transparent;
                    border: none;
                    margin: 0px;
                    padding: 0px;
                }}
                QCheckBox::indicator {{
                    width: 12px;
                    height: 12px;
                    margin: 0px;
                }}
                QCheckBox::indicator:checked {{
                    background-color: {primary};
                    border: 1px solid {bright};
                    border-radius: 3px;
                }}
                QCheckBox::indicator:unchecked {{
                    background-color: #23232c;
                    border: 1px solid #454555;
                    border-radius: 3px;
                }}
                QCheckBox::indicator:checked:disabled {{
                    background-color: #334155;
                    border: 1px solid #475569;
                    border-radius: 3px;
                    image: url("assets/icons/check_ghost.png");
                }}
                QCheckBox::indicator:unchecked:disabled {{
                    background-color: #1a1a22;
                    border: 1px solid #334155;
                    border-radius: 3px;
                }}
            """)
        self.spinbox.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: #1a1a22;
                border: 1px solid #363644;
                border-radius: 4px;
                color: {bright};
                font-size: 10px;
                padding: 0 2px;
            }}
            QDoubleSpinBox:disabled {{
                color: #555562;
                border-color: #262630;
            }}
        """)
        if hasattr(self, 'slider') and self.slider is not None:
            self.slider.set_theme(theme)
        if hasattr(self, 'btn_help') and self.btn_help is not None:
            self.btn_help.set_theme(theme)

    def _val_to_slider(self, fval: float) -> int:
        ratio = (fval - self.min_val) / (self.max_val - self.min_val)
        return int(round(np.clip(ratio, 0.0, 1.0) * self.slider.maximum()))

    def _slider_to_val(self, ival: int) -> float:
        ratio = ival / float(self.slider.maximum())
        val = self.min_val + ratio * (self.max_val - self.min_val)
        return round(val, self.decimals)

    def _on_sub_bypass_changed(self, active: bool):
        self.slider.setEnabled(active)
        self.spinbox.setEnabled(active)
        self.lbl_title.setStyleSheet("color: #b0b0be; font-size: 10px; font-weight: bold;" if active else "color: #777788; font-size: 10px; font-weight: bold;")
        self.bypassToggled.emit()
        self.paramCommitted.emit()

    def _on_slider_changed(self, ival: int):
        fval = self._slider_to_val(ival)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(fval)
        self.spinbox.blockSignals(False)
        if self.is_param_enabled():
            self.valueChanged.emit(fval)

    def _on_spinbox_changed(self, fval: float):
        ival = self._val_to_slider(fval)
        self.slider.blockSignals(True)
        self.slider.setValue(ival)
        self.slider.blockSignals(False)
        if self.is_param_enabled():
            self.valueChanged.emit(fval)

    def reset_to_default(self):
        self.set_value(self.default_val)
        self.paramCommitted.emit()

    def is_param_enabled(self) -> bool:
        if self.has_sub_bypass and self.chk_sub_bypass is not None:
            return self.chk_sub_bypass.isChecked()
        return True

    def set_param_enabled(self, enabled: bool):
        if self.has_sub_bypass and self.chk_sub_bypass is not None:
            self.chk_sub_bypass.setChecked(enabled)
            self.slider.setEnabled(enabled)
            self.spinbox.setEnabled(enabled)
            self.lbl_title.setStyleSheet("color: #b0b0be; font-size: 10px; font-weight: bold;" if enabled else "color: #777788; font-size: 10px; font-weight: bold;")

    def get_effective_value(self) -> float:
        return self.spinbox.value() if self.is_param_enabled() else self.neutral_val

    def get_actual_value(self) -> float:
        return self.spinbox.value()

    def update_title_and_help(self, title: str):
        self.param_title = title
        self.lbl_title.setText(title)
        if hasattr(self, 'btn_help') and self.btn_help is not None:
            self.btn_help.param_title = title

    def set_range(self, min_val: float, max_val: float, step: float = None, default_val: float = None, neutral_val: float = None, decimals: int = None):
        """Динамическое изменение диапазона параметра при смене режима."""
        self.min_val = min_val
        self.max_val = max_val
        if decimals is not None:
            self.decimals = decimals
            self.spinbox.setDecimals(decimals)
        if step is not None:
            self.step = step
            self.spinbox.setSingleStep(step)
        if default_val is not None:
            self.default_val = default_val
        if neutral_val is not None:
            self.neutral_val = neutral_val
        self.spinbox.setRange(min_val, max_val)
        curr = float(np.clip(self.spinbox.value(), min_val, max_val))
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.slider.setValue(self._val_to_slider(curr))
        self.spinbox.setValue(curr)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)

    def set_value(self, val: float):
        val = float(np.clip(val, self.min_val, self.max_val))
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.slider.setValue(self._val_to_slider(val))
        self.spinbox.setValue(val)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)
        if self.is_param_enabled():
            self.valueChanged.emit(val)


class _RotaryDummyLabel:
    def __init__(self, knob):
        self._knob = knob
    def text(self) -> str:
        return f"Wet: {self._knob.value()}%"
    def setStyleSheet(self, s: str):
        pass


class _RotaryDummyButton:
    def __init__(self, knob):
        self._knob = knob
    def click(self):
        self._knob.reset_default()


class RotaryKnob(QWidget):
    """
    Компактная микшерная крутилка (Rotary Knob) в стиле DAW для регулировки Dry/Wet.
    Фиксированный размер: 24x24 px.
    Управление:
      - Вертикальное перетаскивание мыши вверх/вниз с зажатой ЛКМ (0–100%)
      - Вращение колесика мыши (плавная регулировка)
      - Двойной клик ЛКМ: мгновенный сброс на 100%
      - Динамический тултип при наведении и вращении (числовая подпись скрыта по умолчанию)
    """
    valueChanged = pyqtSignal(int)      # 0 .. 100
    mixChanged = pyqtSignal(float)      # 0.0 .. 1.0

    def __init__(self, parent=None, default_val: int = 100):
        super().__init__(parent)
        self.setObjectName("RotaryKnob")
        self.setFixedSize(24, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

        self._val = int(np.clip(default_val, 0, 100))
        self.default_val = self._val
        self._drag_start_y = 0
        self._drag_start_val = 0
        self._dragging = False

        self.lbl_value = _RotaryDummyLabel(self)
        self.btn_reset = _RotaryDummyButton(self)
        self._theme = THEMES['A']
        self._dimmed = False

        self._update_tooltip()

    def set_dimmed(self, dimmed: bool):
        self._dimmed = dimmed
        self.update()

    def set_theme(self, theme: dict):
        self._theme = theme
        self.update()

    def _show_static_tooltip(self):
        point = self.mapToGlobal(QPoint(self.width() // 2, -50))
        text = f"Dry/Wet Mix: {self._val}%\nДвойной клик — сброс на 100%"
        QToolTip.showText(point, text, self)

    def _get_tooltip_text(self) -> str:
        return (
            f"Dry/Wet Mix: {self._val}%\n"
            f"Подмешивание обработанного сигнала к исходному.\n"
            f"Двойной клик — сброс на 100%"
        )

    def _update_tooltip(self):
        self.setToolTip(self._get_tooltip_text())

    def value(self) -> int:
        return self._val

    def ratio(self) -> float:
        return self._val / 100.0

    def set_value(self, val: int):
        val = int(np.clip(val, 0, 100))
        if self._val == val:
            return
        self._val = val
        self._update_tooltip()
        self.update()
        self.valueChanged.emit(self._val)
        self.mixChanged.emit(self._val / 100.0)

    def reset_default(self, *args):
        self.set_value(self.default_val)

    def mousePressEvent(self, event):
        if not self.isEnabled():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_y = event.pos().y()
            self._drag_start_val = self._val
            event.accept()

    def mouseMoveEvent(self, event):
        if not self.isEnabled():
            return
        if self._dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            dy = self._drag_start_y - event.pos().y()
            new_val = int(np.clip(self._drag_start_val + dy, 0, 100))
            if new_val != self._val:
                self.set_value(new_val)
                self._show_static_tooltip()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if not self.isEnabled():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_default()
            self._show_static_tooltip()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if not self.isEnabled():
            return
        delta = event.angleDelta().y()
        if delta != 0:
            step = 2 if delta > 0 else -2
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                step = 1 if delta > 0 else -1
            self.set_value(self._val + step)
            self._show_static_tooltip()
            event.accept()
        else:
            super().wheelEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = float(self.width())
        h = float(self.height())
        cx = w / 2.0
        cy = h / 2.0

        arc_rect = QRectF(2.5, 2.5, w - 5.0, h - 5.0)

        start_angle = 225.0
        total_span = -270.0

        track_pen = QPen(QColor("#282834"), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(arc_rect, int(start_angle * 16), int(total_span * 16))

        t = self._val / 100.0
        active_span = total_span * t
        primary_col = self._theme.get('primary', '#00adb5') if hasattr(self, '_theme') and self._theme else '#00adb5'
        bright_col = self._theme.get('bright', '#00fff2') if hasattr(self, '_theme') and self._theme else '#00fff2'

        if abs(active_span) > 0.1:
            if not self.isEnabled() or getattr(self, '_dimmed', False):
                active_color = QColor("#475569")
            elif self._val == 100:
                active_color = QColor(bright_col)
            else:
                active_color = QColor(primary_col)
            active_pen = QPen(active_color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(active_pen)
            painter.drawArc(arc_rect, int(start_angle * 16), int(active_span * 16))

        body_r = 6.8
        body_rect = QRectF(cx - body_r, cy - body_r, body_r * 2.0, body_r * 2.0)
        is_active = self.isEnabled() and not getattr(self, '_dimmed', False)
        body_color = QColor("#1e1e28") if is_active else QColor("#16161e")
        border_color = QColor("#363646") if is_active else QColor("#262630")
        painter.setPen(QPen(border_color, 1.0))
        painter.setBrush(QBrush(body_color))
        painter.drawEllipse(body_rect)

        cur_angle_deg = start_angle + active_span
        rad = np.radians(cur_angle_deg)
        r_inner = 1.5
        r_outer = 6.0
        p1 = QPointF(cx + r_inner * np.cos(rad), cy - r_inner * np.sin(rad))
        p2 = QPointF(cx + r_outer * np.cos(rad), cy - r_outer * np.sin(rad))

        pointer_color = QColor(bright_col) if is_active else QColor("#475569")
        pointer_pen = QPen(pointer_color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pointer_pen)
        painter.drawLine(p1, p2)


DryWetControl = RotaryKnob


class StateCopyButton(QPushButton):
    """Кнопка копирования стейтов с двухцветной маркировкой [ A → B ] или [ B → A ]."""

    def __init__(self, from_state: str, to_state: str, parent=None):
        super().__init__(f"{from_state} → {to_state}", parent)
        self._from_state = from_state.upper()
        self._to_state = to_state.upper()
        self.setFixedHeight(24)
        self.setFixedWidth(54)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #1e1e28;
                border: 1px solid #363648;
                border-radius: 4px;
                padding: 0 4px;
                color: transparent;
            }
            QPushButton:hover {
                background-color: #262634;
                border-color: #4a4a5e;
            }
            QPushButton:pressed {
                background-color: #14141c;
            }
            QPushButton:disabled {
                background-color: #141418;
                border-color: #22222a;
            }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()

        col_a = QColor("#00E5FF") if self.isEnabled() else QColor("#2d5358")
        col_b = QColor("#FF9E0B") if self.isEnabled() else QColor("#59411d")
        col_arrow = QColor("#E2E8F0") if self.isEnabled() else QColor("#4b4b55")

        c1 = col_a if self._from_state == "A" else col_b
        c2 = col_b if self._to_state == "B" else col_a

        t1 = self._from_state
        t_arrow = " → "
        t2 = self._to_state

        w1 = fm.horizontalAdvance(t1)
        w_arrow = fm.horizontalAdvance(t_arrow)
        w2 = fm.horizontalAdvance(t2)
        total_w = w1 + w_arrow + w2

        start_x = (self.width() - total_w) / 2.0
        y = (self.height() + fm.ascent() - fm.descent()) / 2.0

        # 1. Первая буква
        painter.setPen(c1)
        painter.drawText(int(start_x), int(y), t1)

        # 2. Стрелка
        painter.setPen(col_arrow)
        painter.drawText(int(start_x + w1), int(y), t_arrow)

        # 3. Вторая буква
        painter.setPen(c2)
        painter.drawText(int(start_x + w1 + w_arrow), int(y), t2)


class _DryWetRowAdapter(QObject):
    valueChanged = pyqtSignal(float)
    bypassToggled = pyqtSignal()
    paramCommitted = pyqtSignal()

    def __init__(self, ctrl: DryWetControl):
        super().__init__()
        self._ctrl = ctrl
        if ctrl is not None:
            ctrl.valueChanged.connect(lambda v: self.valueChanged.emit(float(v)))
            ctrl.valueChanged.connect(lambda _: self.paramCommitted.emit())

    def get_actual_value(self) -> float:
        return float(self._ctrl.value()) if self._ctrl is not None else 100.0

    def get_effective_value(self) -> float:
        return float(self._ctrl.value()) if self._ctrl is not None else 100.0

    def set_value(self, val: float):
        if self._ctrl is not None:
            ival = int(round(val if val > 1.0 else val * 100.0))
            self._ctrl.set_value(ival)

    def is_param_enabled(self) -> bool:
        return self._ctrl.isEnabled() if self._ctrl is not None else True

    def set_param_enabled(self, enabled: bool):
        if self._ctrl is not None:
            self._ctrl.setEnabled(enabled)


class ElidedLabel(QLabel):
    """QLabel с эластичной минимальной шириной, не распирающий горизонтальный layout."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._full_text = text

    def setText(self, text):
        self._full_text = str(text)
        super().setText(text)

    def text(self) -> str:
        return getattr(self, '_full_text', super().text())

    def minimumSizeHint(self) -> QSize:
        return QSize(20, self.fontMetrics().height())


class CollapsibleHeader(QFrame):
    toggle_requested = pyqtSignal()

    def __init__(self, title: str, has_dry_wet: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("CollapsibleHeader")
        self.setFixedHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme = THEMES['A']

        primary = self._theme['primary']
        bright = self._theme['bright']

        self.setStyleSheet(f"""
            QFrame#CollapsibleHeader {{
                background-color: #141419;
                border: 1px solid #282834;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }}
            QFrame#CollapsibleHeader:hover {{
                border-color: {primary};
            }}
            QFrame#CollapsibleHeader QLabel {{
                background: transparent;
                border: none;
            }}
            QFrame#CollapsibleHeader QCheckBox {{
                background: transparent;
                border: none;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(8)

        self.chk_bypass = QCheckBox()
        self.chk_bypass.setChecked(False)
        self.chk_bypass.setFixedSize(16, 16)
        self.chk_bypass.setToolTip("Включить / Обойти (Bypass) данный модуль")
        self.chk_bypass.setStyleSheet(f"""
            QCheckBox {{
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }}
            QCheckBox::indicator {{
                width: 13px;
                height: 13px;
                margin: 0px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {primary};
                border: 1px solid {bright};
                border-radius: 4px;
            }}
            QCheckBox::indicator:unchecked {{
                background-color: #2b2b36;
                border: 1px solid #4a4a58;
                border-radius: 4px;
            }}
            QCheckBox::indicator:checked:disabled {{
                background-color: #334155;
                border: 1px solid #475569;
                border-radius: 4px;
                image: url("assets/icons/check_ghost.png");
            }}
            QCheckBox::indicator:unchecked:disabled {{
                background-color: #1a1a22;
                border: 1px solid #334155;
                border-radius: 4px;
            }}
        """)

        self.arrow_btn = QPushButton("▼")
        self.arrow_btn.setFixedSize(16, 16)
        self.arrow_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {primary};
                font-weight: bold;
                font-size: 10px;
            }}
        """)

        self.lbl_title = ElidedLabel(title, self)
        self.lbl_title.setStyleSheet("background: transparent; color: #b0b0c0; font-size: 12px; font-weight: bold;")
        self.lbl_title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        layout.addWidget(self.chk_bypass)
        layout.addWidget(self.arrow_btn)
        layout.addWidget(self.lbl_title)
        layout.addStretch()

        if has_dry_wet:
            self.dry_wet_ctrl = DryWetControl(self)
            layout.addWidget(self.dry_wet_ctrl)
        else:
            self.dry_wet_ctrl = None

    def apply_theme(self, theme: dict):
        self._theme = theme
        primary = theme['primary']
        bright = theme['bright']
        self.arrow_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {primary};
                font-weight: bold;
                font-size: 10px;
            }}
        """)
        self.chk_bypass.setStyleSheet(f"""
            QCheckBox {{
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }}
            QCheckBox::indicator {{
                width: 13px;
                height: 13px;
                margin: 0px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {primary};
                border: 1px solid {bright};
                border-radius: 4px;
            }}
            QCheckBox::indicator:unchecked {{
                background-color: #2b2b36;
                border: 1px solid #4a4a58;
                border-radius: 4px;
            }}
            QCheckBox::indicator:checked:disabled {{
                background-color: #334155;
                border: 1px solid #475569;
                border-radius: 4px;
                image: url("assets/icons/check_ghost.png");
            }}
            QCheckBox::indicator:unchecked:disabled {{
                background-color: #1a1a22;
                border: 1px solid #334155;
                border-radius: 4px;
            }}
        """)
        is_content_vis = False
        if self.parent() and hasattr(self.parent(), 'content_widget'):
            is_content_vis = self.parent().content_widget.isVisible()
        self.set_corner_rounding(is_content_vis)
        self._update_title_color()
        if hasattr(self, 'dry_wet_ctrl') and self.dry_wet_ctrl is not None:
            self.dry_wet_ctrl.set_theme(theme)

    def _update_title_color(self):
        primary = self._theme.get('primary', '#00adb5') if hasattr(self, '_theme') and self._theme else '#00adb5'
        if getattr(self, '_is_master_bypassed', False):
            self.lbl_title.setStyleSheet("background: transparent; color: #64748B; font-size: 10.5px; font-weight: bold;")
            self.arrow_btn.setStyleSheet("background: transparent; border: none; color: #64748B; font-weight: bold; font-size: 10px;")
        elif self.chk_bypass.isChecked():
            self.lbl_title.setStyleSheet(f"background: transparent; color: {primary}; font-size: 10.5px; font-weight: bold;")
            self.arrow_btn.setStyleSheet(f"background: transparent; border: none; color: {primary}; font-weight: bold; font-size: 10px;")
        else:
            self.lbl_title.setStyleSheet("background: transparent; color: #64748B; font-size: 10.5px; font-weight: bold;")
            self.arrow_btn.setStyleSheet("background: transparent; border: none; color: #64748B; font-weight: bold; font-size: 10px;")

    def set_corner_rounding(self, content_visible: bool):
        primary = self._theme.get('primary', '#00adb5') if hasattr(self, '_theme') and self._theme else '#00adb5'
        if content_visible:
            self.setStyleSheet(f"""
                QFrame#CollapsibleHeader {{
                    background-color: #141419;
                    border: 1px solid #282834;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    border-bottom-left-radius: 0px;
                    border-bottom-right-radius: 0px;
                }}
                QFrame#CollapsibleHeader:hover {{
                    border-color: {primary};
                }}
                QFrame#CollapsibleHeader QLabel {{
                    background: transparent;
                    border: none;
                }}
                QFrame#CollapsibleHeader QCheckBox {{
                    background: transparent;
                    border: none;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame#CollapsibleHeader {{
                    background-color: #141419;
                    border: 1px solid #282834;
                    border-radius: 8px;
                }}
                QFrame#CollapsibleHeader:hover {{
                    border-color: {primary};
                }}
                QFrame#CollapsibleHeader QLabel {{
                    background: transparent;
                    border: none;
                }}
                QFrame#CollapsibleHeader QCheckBox {{
                    background: transparent;
                    border: none;
                }}
            """)

    def mousePressEvent(self, event):
        child = self.childAt(event.pos())
        if child is self.chk_bypass or child is self.arrow_btn:
            super().mousePressEvent(event)
            return
        if self.dry_wet_ctrl is not None and (child is self.dry_wet_ctrl or self.dry_wet_ctrl.isAncestorOf(child)):
            super().mousePressEvent(event)
            return
        self.toggle_requested.emit()
        event.accept()

    def set_master_bypassed(self, bypassed: bool):
        self._is_master_bypassed = bypassed
        is_vis = self.parent().content_widget.isVisible() if self.parent() and hasattr(self.parent(), 'content_widget') else False
        if bypassed:
            rounding_style = """
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            """ if is_vis else "border-radius: 8px;"
            self.setStyleSheet(f"""
                QFrame#CollapsibleHeader {{
                    background-color: #121216;
                    border: 1px solid #1E293B;
                    {rounding_style}
                }}
                QFrame#CollapsibleHeader QLabel {{
                    background: transparent;
                    border: none;
                }}
                QFrame#CollapsibleHeader QCheckBox {{
                    background: transparent;
                    border: none;
                }}
            """)
            self.lbl_title.setStyleSheet("background: transparent; color: #64748B; font-size: 10.5px; font-weight: bold;")
            self.arrow_btn.setStyleSheet("background: transparent; border: none; color: #64748B;")
        else:
            self._update_title_color()
            self.set_corner_rounding(is_vis)
            primary = self._theme.get('primary', '#00adb5') if hasattr(self, '_theme') and self._theme else '#00adb5'
            self.arrow_btn.setStyleSheet(f"background: transparent; border: none; color: {primary}; font-weight: bold; font-size: 10px;")

    def mouseDoubleClickEvent(self, event):
        child = self.childAt(event.pos())
        if self.dry_wet_ctrl is not None and (child is self.dry_wet_ctrl or self.dry_wet_ctrl.isAncestorOf(child)):
            self.dry_wet_ctrl.reset_default()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class CollapsibleInsertBox(QWidget):
    slider_changed = pyqtSignal()
    bypass_toggled = pyqtSignal()
    state_committed = pyqtSignal()
    dry_wet_changed = pyqtSignal(int)

    def __init__(self, title: str, has_dry_wet: bool = False, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = CollapsibleHeader(title, has_dry_wet=has_dry_wet, parent=self)
        self.header.toggle_requested.connect(self._toggle_collapse)
        self.header.arrow_btn.clicked.connect(self._toggle_collapse)
        self.header.chk_bypass.toggled.connect(self._on_header_bypass_toggled)

        if self.header.dry_wet_ctrl is not None:
            self.header.dry_wet_ctrl.valueChanged.connect(self._on_dry_wet_value_changed)

        self.content_widget = QFrame()
        self.content_widget.setObjectName("InsertContent")
        self.content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.content_widget.setStyleSheet("""
            QFrame#InsertContent {
                background-color: #141419;
                border: 1px solid #23232c;
                border-top: none;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QFrame#InsertContent:disabled {
                background-color: #0f0f13;
                border-color: #1a1a20;
            }
        """)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(8)

        layout.addWidget(self.header)
        layout.addWidget(self.content_widget)

        # По умолчанию модуль деактивирован (bypass): блокируем все параметры
        self.content_widget.setEnabled(False)
        self._update_header_style(False)

    def _on_dry_wet_value_changed(self, val: int):
        self.dry_wet_changed.emit(val)
        self.slider_changed.emit()
        self.state_committed.emit()

    def get_dry_wet_ratio(self) -> float:
        if self.header.dry_wet_ctrl is not None:
            return self.header.dry_wet_ctrl.ratio()
        return 1.0

    def set_dry_wet_ratio(self, ratio: float):
        if self.header.dry_wet_ctrl is not None:
            val = int(round(ratio if ratio > 1.0 else ratio * 100.0))
            self.header.dry_wet_ctrl.set_value(val)

    def _on_header_bypass_toggled(self, checked: bool = False):
        active = self.is_enabled()
        self.content_widget.setEnabled(active)
        self._update_header_style(active)
        self.bypass_toggled.emit()
        self.state_committed.emit()

    def apply_theme(self, theme: dict):
        self._theme = theme
        self.header.apply_theme(theme)
        for row in self.findChildren(AstroParamRow):
            row.apply_theme(theme)
        for icon in self.findChildren(HelpIcon):
            icon.set_theme(theme)
        primary = theme['primary']
        bright = theme['bright']
        for chk in self.findChildren(QCheckBox):
            if chk is not self.header.chk_bypass and not (chk.parent() and hasattr(chk.parent(), 'chk_sub_bypass') and chk is chk.parent().chk_sub_bypass):
                chk.setStyleSheet(f"""
                    QCheckBox {{
                        background: transparent;
                        color: #b0b0be;
                        font-size: 11px;
                    }}
                    QCheckBox::indicator {{
                        width: 13px;
                        height: 13px;
                        margin: 0px;
                    }}
                    QCheckBox::indicator:checked {{
                        background-color: {primary};
                        border: 1px solid {bright};
                        border-radius: 4px;
                    }}
                    QCheckBox::indicator:unchecked {{
                        background-color: #23232c;
                        border: 1px solid #454555;
                        border-radius: 4px;
                    }}
                    QCheckBox::indicator:checked:disabled {{
                        background-color: #334155;
                        border: 1px solid #475569;
                        border-radius: 4px;
                        image: url("assets/icons/check_ghost.png");
                    }}
                    QCheckBox::indicator:unchecked:disabled {{
                        background-color: #1a1a22;
                        border: 1px solid #334155;
                        border-radius: 4px;
                    }}
                """)
        self._update_header_style(self.is_enabled() and not getattr(self, '_is_master_bypassed', False))
        self.update()

    def _set_controls_enabled(self, enabled: bool):
        """Обновляет интерактивность и стиль всех контролов инсерта при локальном или мастер-байпасе."""
        for cb in self.findChildren(QComboBox):
            cb.setEnabled(enabled)
        for icon in self.findChildren(HelpIcon):
            icon.setEnabled(enabled)
        for row in self.findChildren(AstroParamRow):
            if hasattr(row, 'btn_help') and row.btn_help is not None:
                row.btn_help.setEnabled(enabled)
            if hasattr(row, 'btn_reset') and row.btn_reset is not None:
                row.btn_reset.setEnabled(enabled)
            if hasattr(row, 'chk_sub_bypass') and row.chk_sub_bypass is not None:
                row.chk_sub_bypass.setEnabled(enabled)
            if not enabled:
                row.slider.setEnabled(False)
                row.spinbox.setEnabled(False)
                if hasattr(row, 'lbl_title') and row.lbl_title is not None:
                    row.lbl_title.setStyleSheet("color: #64748B; font-size: 10px; font-weight: bold;")
            else:
                param_on = row.is_param_enabled()
                row.slider.setEnabled(param_on)
                row.spinbox.setEnabled(param_on)
                if hasattr(row, 'lbl_title') and row.lbl_title is not None:
                    row.lbl_title.setStyleSheet("color: #b0b0be; font-size: 10px; font-weight: bold;" if param_on else "color: #777788; font-size: 10px; font-weight: bold;")
        for btn in self.content_widget.findChildren(QPushButton):
            btn.setEnabled(enabled)
        for chk in self.content_widget.findChildren(QCheckBox):
            if not (chk.parent() and hasattr(chk.parent(), 'chk_sub_bypass') and chk is chk.parent().chk_sub_bypass):
                chk.setEnabled(enabled)
        if hasattr(self, '_update_mode_controls_state'):
            self._update_mode_controls_state(enabled)

    def set_master_bypassed(self, bypassed: bool):
        """Визуальный отклик на Master Bypass: деактивация элементов, приглушение крутилок и рамок."""
        self._is_master_bypassed = bypassed
        if bypassed:
            self.content_widget.setEnabled(False)
            self.header.chk_bypass.setEnabled(False)
            if self.header.dry_wet_ctrl is not None:
                self.header.dry_wet_ctrl.set_dimmed(True)
                self.header.dry_wet_ctrl.setEnabled(False)
            self.header.set_master_bypassed(True)
            self._set_controls_enabled(False)
            self.content_widget.setStyleSheet("""
                QFrame#InsertContent {
                    background-color: #121216;
                    border: 1px solid #1E293B;
                    border-top: none;
                    border-bottom-left-radius: 8px;
                    border-bottom-right-radius: 8px;
                }
            """)
        else:
            self.header.chk_bypass.setEnabled(True)
            active = self.is_enabled()
            self.content_widget.setEnabled(active)
            if self.header.dry_wet_ctrl is not None:
                self.header.dry_wet_ctrl.set_dimmed(not active)
                self.header.dry_wet_ctrl.setEnabled(active)
            self.header.set_master_bypassed(False)
            self.content_widget.setStyleSheet("""
                QFrame#InsertContent {
                    background-color: #141419;
                    border: 1px solid #23232c;
                    border-top: none;
                    border-bottom-left-radius: 8px;
                    border-bottom-right-radius: 8px;
                }
                QFrame#InsertContent:disabled {
                    background-color: #0f0f13;
                    border-color: #1a1a20;
                }
            """)
            self._update_header_style(active)
            self.header.set_corner_rounding(self.content_widget.isVisible())

    def _update_header_style(self, enabled: bool):
        self.header._update_title_color()
        if self.header.dry_wet_ctrl is not None:
            self.header.dry_wet_ctrl.setEnabled(enabled)
            self.header.dry_wet_ctrl.set_dimmed(not enabled)
        self._set_controls_enabled(enabled)

    def _toggle_collapse(self):
        is_vis = self.content_widget.isVisible()
        self.content_widget.setVisible(not is_vis)
        self.header.arrow_btn.setText("►" if is_vis else "▼")
        self.header.set_corner_rounding(not is_vis)

    def is_enabled(self) -> bool:
        return self.header.chk_bypass.isChecked()

    def set_enabled(self, val: bool):
        self.header.chk_bypass.setChecked(val)
        self.content_widget.setEnabled(val)
        self._update_header_style(val)


# =============================================================================
# МОДУЛИ ИНСЕРТОВ РЭКА
# =============================================================================

WAVELET_MAX_LAYERS = 6
WAVELET_MIN_LAYERS = 1
WAVELET_DEFAULT_LAYERS = 2

_MINI_BTN_STYLE = """
    QPushButton {
        background-color: #1a1a22;
        color: #b0b0be;
        border: 1px solid #363644;
        border-radius: 3px;
        font-size: 13px;
        font-weight: bold;
        padding: 0;
    }
    QPushButton:hover {
        background-color: #2a2a36;
        color: #00fff2;
        border-color: #00adb5;
    }
    QPushButton:pressed {
        background-color: #00adb5;
        color: #111114;
    }
    QPushButton:disabled {
        background-color: #14141a;
        color: #44444f;
        border-color: #22222a;
    }
"""


def _layer_detail_scale(n: int) -> str:
    """Масштаб деталей слоя N вейвлет-разложения à trous (в пикселях)."""
    return f"{2 ** (n - 1)}–{2 ** n} px"


class WaveletLayerWidget(QFrame):
    """Компактная карточка одного слоя вейвлета: байпас, Amount, Denoise и сброс [R]."""
    changed = pyqtSignal()      # live-изменение значения (рендер)
    bypassChanged = pyqtSignal()
    committed = pyqtSignal()    # фиксация значения (история)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.setObjectName("WaveletLayer")
        self.setStyleSheet("""
            QFrame#WaveletLayer {
                background-color: transparent;
                border: none;
                border-bottom: 1px solid #25252f;
            }
        """)

        help_key = f"Wavelet Layer {index}"
        ASTRO_HELP_TEXTS.setdefault(help_key, (
            f"<b>Слой {index} вейвлет-разложения (à trous, B3-сплайн)</b><br><br>"
            f"Отвечает за детали масштаба примерно {_layer_detail_scale(index)}. "
            "Чем выше номер слоя, тем более крупные структуры он усиливает. "
            "Amount = 1.00 — нейтраль. Denoise — порог мягкого подавления шума внутри этого слоя "
            "(0 — фильтрация пропускается)."
        ))

        max_amount = 10.0 if index == 1 else 8.0
        self.row_amount = AstroParamRow(
            title=f"Layer {index} ({_layer_detail_scale(index)} details)",
            min_val=1.00, max_val=max_amount, default_val=1.00, step=0.05, decimals=2,
            neutral_val=1.00, has_sub_bypass=True
        )
        self.row_amount.btn_help.param_title = help_key
        self.row_denoise = AstroParamRow(
            title="Denoise",
            min_val=0.00, max_val=0.50, default_val=0.00, step=0.005, decimals=3,
            neutral_val=0.00, has_sub_bypass=True
        )
        # Компактный подползунок: свой тумблер байпаса и [R] (сброс в 0.000), без справки
        self.row_denoise.btn_help.hide()
        self.row_denoise.btn_reset.setToolTip("Сбросить Denoise в 0.000")
        self.row_denoise.lbl_title.setStyleSheet("color: #8a8a99; font-size: 9px; font-weight: bold;")
        self.row_denoise.chk_sub_bypass.setToolTip("Включить / отключить шумоподавление этого слоя")
        self.row_denoise.layout().setContentsMargins(0, 0, 0, 2)

        # Кнопка сброса всего слоя (Amount + Denoise)
        try:
            self.row_amount.btn_reset.clicked.disconnect()
        except TypeError:
            pass
        self.row_amount.btn_reset.setToolTip("Сбросить слой (Amount и Denoise) к умолчанию")
        self.row_amount.btn_reset.clicked.connect(self.reset_layer)
        try:
            self.row_amount.slider.doubleClicked.disconnect()
        except TypeError:
            pass
        self.row_amount.slider.doubleClicked.connect(self.reset_layer)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 4)
        lay.setSpacing(4)
        lay.addWidget(self.row_amount)
        lay.addWidget(self.row_denoise)

        self.row_denoise.setEnabled(False)
        self.row_amount.bypassToggled.connect(self._on_layer_bypass)
        self.row_denoise.bypassToggled.connect(lambda: self.bypassChanged.emit())
        self.row_amount.valueChanged.connect(lambda _: self.changed.emit())
        self.row_denoise.valueChanged.connect(lambda _: self.changed.emit())
        self.row_amount.paramCommitted.connect(self._emit_committed)
        self.row_denoise.paramCommitted.connect(self._emit_committed)

    def _emit_committed(self):
        self.committed.emit()

    def _on_layer_bypass(self):
        self.row_denoise.setEnabled(self.is_layer_enabled())
        self.bypassChanged.emit()

    def is_layer_enabled(self) -> bool:
        return self.row_amount.is_param_enabled()

    def reset_layer(self):
        self.row_amount.set_value(self.row_amount.default_val)
        self.row_denoise.set_value(self.row_denoise.default_val)
        self.changed.emit()
        self.committed.emit()

    def get_config(self) -> dict:
        enabled = self.is_layer_enabled()
        denoise_on = self.row_denoise.is_param_enabled()
        raw_amount = float(self.row_amount.get_actual_value())
        raw_denoise = float(self.row_denoise.get_actual_value())
        return {
            "amount": raw_amount if enabled else 1.0,
            "denoise": raw_denoise if (enabled and denoise_on) else 0.0,
            "enabled": enabled,
            "denoise_on": denoise_on,
            "raw_amount": raw_amount,
            "raw_denoise": raw_denoise,
        }

    def set_config(self, cfg: dict):
        raw_denoise = float(cfg.get("raw_denoise", cfg.get("denoise", 0.0)))
        self.row_amount.set_value(cfg.get("raw_amount", cfg.get("amount", 1.0)))
        self.row_denoise.set_value(raw_denoise)
        enabled = bool(cfg.get("enabled", False))
        denoise_on = bool(cfg.get("denoise_on", raw_denoise > 0.0))
        self.row_amount.set_param_enabled(enabled)
        self.row_denoise.set_param_enabled(denoise_on)
        self.row_denoise.setEnabled(enabled)


class WaveletInsert(CollapsibleInsertBox):
    layers_changed = pyqtSignal(str)   # имя действия: "Add Layer" / "Remove Layer"

    def __init__(self, parent=None):
        super().__init__("INSERT 1: Sharpness (Wavelet / Unsharp)", has_dry_wet=True, parent=parent)
        self.row_wet_mix = _DryWetRowAdapter(self.header.dry_wet_ctrl)

        mode_hbox = QHBoxLayout()
        mode_hbox.setContentsMargins(0, 0, 0, 0)
        mode_hbox.setSpacing(6)
        mode_hbox.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.lbl_mode = QLabel("Mode:")
        self.lbl_mode.setStyleSheet("color: #00adb5; font-size: 10px; font-weight: bold;")
        self.combo_mode = QComboBox()
        self.combo_mode.setFixedHeight(22)
        self.combo_mode.addItems(["Wavelet (à trous)", "Unsharp Mask"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_help_mode = HelpIcon("Sharpness Mode")
        mode_hbox.addWidget(self.lbl_mode)
        mode_hbox.addWidget(self.combo_mode)
        mode_hbox.addWidget(self.btn_help_mode)
        mode_hbox.addStretch()
        self.content_layout.addLayout(mode_hbox)

        # --- Режим Wavelet: динамический стек слоёв -------------------------------------
        self.wavelet_panel = QWidget()
        self.wavelet_vbox = QVBoxLayout(self.wavelet_panel)
        self.wavelet_vbox.setContentsMargins(0, 0, 0, 0)
        self.wavelet_vbox.setSpacing(8)

        self.layers_container = QVBoxLayout()
        self.layers_container.setContentsMargins(0, 0, 0, 0)
        self.layers_container.setSpacing(6)
        self.wavelet_vbox.addLayout(self.layers_container)
        self.layer_widgets = []

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.setSpacing(6)
        self.btn_add_layer = QPushButton("+")
        self.btn_add_layer.setFixedSize(26, 20)
        self.btn_add_layer.setToolTip("Add Layer — добавить следующий уровень деталей")
        self.btn_add_layer.setStyleSheet(_MINI_BTN_STYLE)
        self.btn_add_layer.clicked.connect(lambda: self.add_layer(user_action=True))
        self.btn_remove_layer = QPushButton("−")
        self.btn_remove_layer.setFixedSize(26, 20)
        self.btn_remove_layer.setToolTip("Remove Layer — удалить последний слой")
        self.btn_remove_layer.setStyleSheet(_MINI_BTN_STYLE)
        self.btn_remove_layer.clicked.connect(lambda: self.remove_layer(user_action=True))
        self.lbl_layer_count = QLabel("")
        self.lbl_layer_count.setStyleSheet("color: #666675; font-size: 9px;")
        ctrl.addWidget(self.btn_add_layer)
        ctrl.addWidget(self.btn_remove_layer)
        ctrl.addWidget(self.lbl_layer_count)
        ctrl.addStretch()
        self.wavelet_vbox.addLayout(ctrl)
        self.content_layout.addWidget(self.wavelet_panel)

        # --- Режим Unsharp Mask: прежние три строки ------------------------------------
        self.unsharp_panel = QWidget()
        unsharp_vbox = QVBoxLayout(self.unsharp_panel)
        unsharp_vbox.setContentsMargins(0, 0, 0, 0)
        unsharp_vbox.setSpacing(8)

        self.row_l1 = AstroParamRow(
            title="Amount (Strength)",
            min_val=1.00, max_val=15.00, default_val=1.00, step=0.10, decimals=2, neutral_val=1.00, has_sub_bypass=False
        )
        self.row_l2 = AstroParamRow(
            title="Radius (Sigma Blur)",
            min_val=0.50, max_val=20.00, default_val=1.50, step=0.20, decimals=2, neutral_val=1.00, has_sub_bypass=False
        )
        self.row_denoise = AstroParamRow(
            title="Threshold (Noise Floor)",
            min_val=0.00, max_val=0.20, default_val=0.00, step=0.005, decimals=3, neutral_val=0.00, has_sub_bypass=False
        )
        self.row_l1.btn_help.param_title = "Amount (Strength)"
        self.row_l2.btn_help.param_title = "Radius (Sigma Blur)"
        self.row_denoise.btn_help.param_title = "Threshold (Noise Floor)"

        for row in [self.row_l1, self.row_l2, self.row_denoise]:
            row.valueChanged.connect(lambda _: self.slider_changed.emit())
            row.bypassToggled.connect(lambda: self.bypass_toggled.emit())
            row.paramCommitted.connect(lambda: self.state_committed.emit())
            unsharp_vbox.addWidget(row)
        self.content_layout.addWidget(self.unsharp_panel)

        for _ in range(WAVELET_DEFAULT_LAYERS):
            self.add_layer()
        self._update_mode_visibility()
        self._update_header_style(False)

    def _update_mode_controls_state(self, enabled: bool):
        primary = self._theme.get('primary', '#00adb5') if hasattr(self, '_theme') and self._theme else '#00adb5'
        if hasattr(self, 'lbl_mode') and self.lbl_mode is not None:
            self.lbl_mode.setStyleSheet(f"color: {primary if enabled else '#64748B'}; font-size: 10px; font-weight: bold;")
        if hasattr(self, 'combo_mode') and self.combo_mode is not None:
            self.combo_mode.setEnabled(enabled)
        if hasattr(self, 'btn_help_mode') and self.btn_help_mode is not None:
            self.btn_help_mode.setEnabled(enabled)
        if hasattr(self, 'btn_add_layer') and self.btn_add_layer is not None:
            self.btn_add_layer.setEnabled(enabled and self.layer_count() < WAVELET_MAX_LAYERS)
        if hasattr(self, 'btn_remove_layer') and self.btn_remove_layer is not None:
            self.btn_remove_layer.setEnabled(enabled and self.layer_count() > WAVELET_MIN_LAYERS)

    # --- Управление стеком слоёв -----------------------------------------------------
    def layer_count(self) -> int:
        return len(self.layer_widgets)

    def add_layer(self, user_action: bool = False) -> bool:
        if self.layer_count() >= WAVELET_MAX_LAYERS:
            return False
        w = WaveletLayerWidget(self.layer_count() + 1)
        w.changed.connect(self.slider_changed.emit)
        w.bypassChanged.connect(self.bypass_toggled.emit)
        w.committed.connect(self.state_committed.emit)
        self.layers_container.addWidget(w)
        self.layer_widgets.append(w)
        self._update_layer_buttons()
        if user_action:
            self.layers_changed.emit("Add Layer")
        return True

    def remove_layer(self, user_action: bool = False) -> bool:
        if self.layer_count() <= WAVELET_MIN_LAYERS:
            return False
        w = self.layer_widgets.pop()
        self.layers_container.removeWidget(w)
        w.hide()
        w.deleteLater()
        self._update_layer_buttons()
        if user_action:
            self.layers_changed.emit("Remove Layer")
        return True

    def set_layer_count(self, n: int):
        n = max(WAVELET_MIN_LAYERS, min(WAVELET_MAX_LAYERS, int(n)))
        while self.layer_count() < n:
            self.add_layer()
        while self.layer_count() > n:
            self.remove_layer()

    def _update_layer_buttons(self):
        n = self.layer_count()
        self.btn_add_layer.setEnabled(n < WAVELET_MAX_LAYERS)
        self.btn_remove_layer.setEnabled(n > WAVELET_MIN_LAYERS)
        self.lbl_layer_count.setText(f"{n} / {WAVELET_MAX_LAYERS} layers")

    def _update_mode_visibility(self):
        wavelet = self.combo_mode.currentIndex() == 0
        self.wavelet_panel.setVisible(wavelet)
        self.unsharp_panel.setVisible(not wavelet)

    def set_flat(self):
        """Возврат к стартовому состоянию: 2 слоя, всё выключено, значения по умолчанию."""
        self.set_layer_count(WAVELET_DEFAULT_LAYERS)
        for w in self.layer_widgets:
            w.set_config({"raw_amount": 1.0, "raw_denoise": 0.0, "enabled": False, "denoise_on": False})
        for row in (self.row_l1, self.row_l2, self.row_denoise):
            row.set_value(row.default_val)
            row.set_param_enabled(False)
        self.set_dry_wet_ratio(100.0)

    def _on_mode_changed(self, idx: int):
        self._update_mode_visibility()
        self.slider_changed.emit()
        self.state_committed.emit()

    def get_values(self):
        layers = [w.get_config() for w in self.layer_widgets]
        wet_val = float(self.get_dry_wet_ratio() * 100.0)
        return {
            'w_mode': self.combo_mode.currentIndex(),
            'layers_config': layers,
            'w_l1': self.row_l1.get_effective_value(),
            'w_l2': self.row_l2.get_effective_value(),
            'w_denoise': self.row_denoise.get_effective_value(),
            'w_l1_on': self.row_l1.is_param_enabled(),
            'w_l2_on': self.row_l2.is_param_enabled(),
            'w_denoise_on': self.row_denoise.is_param_enabled(),
            'w_wet_mix': wet_val,
            'insert1_wet_mix': wet_val,
        }

    def set_values(self, d: dict):
        mode = d.get('w_mode', 0)
        self.combo_mode.blockSignals(True)
        self.combo_mode.setCurrentIndex(mode)
        self.combo_mode.blockSignals(False)
        self._update_mode_visibility()

        if mode == 0:
            cfgs = d.get('layers_config')
            if not cfgs:
                # Обратная совместимость: старый пресет из 2 слоёв (w_l1/w_l2/w_denoise)
                cfgs = [
                    {"raw_amount": d.get('w_l1', 1.0), "raw_denoise": d.get('w_denoise', 0.0),
                     "enabled": d.get('w_l1_on', False), "denoise_on": d.get('w_denoise_on', False)},
                    {"raw_amount": d.get('w_l2', 1.0), "raw_denoise": 0.0,
                     "enabled": d.get('w_l2_on', False), "denoise_on": False},
                ]
            self.set_layer_count(len(cfgs))
            for w, cfg in zip(self.layer_widgets, cfgs):
                w.set_config(cfg)
            for row in (self.row_l1, self.row_l2, self.row_denoise):
                row.set_value(row.default_val)
                row.set_param_enabled(False)
        else:
            if 'layers_config' in d and d['layers_config']:
                self.set_layer_count(len(d['layers_config']))
                for w, cfg in zip(self.layer_widgets, d['layers_config']):
                    w.set_config(cfg)
            self.row_l1.set_value(d.get('w_l1', 1.0))
            self.row_l2.set_value(d.get('w_l2', 1.5))
            self.row_denoise.set_value(d.get('w_denoise', 0.0))
            self.row_l1.set_param_enabled(d.get('w_l1_on', False))
            self.row_l2.set_param_enabled(d.get('w_l2_on', False))
            self.row_denoise.set_param_enabled(d.get('w_denoise_on', False))
        self.set_dry_wet_ratio(float(d.get('w_wet_mix', d.get('insert1_wet_mix', d.get('wet_mix', 100.0)))))


class RichardsonLucyInsert(CollapsibleInsertBox):
    def __init__(self, parent=None):
        super().__init__("INSERT 2: Richardson-Lucy Deconvolution", has_dry_wet=True, parent=parent)
        self.row_wet_mix = _DryWetRowAdapter(self.header.dry_wet_ctrl)

        self.row_sigma = AstroParamRow(
            title="Radius (Sigma PSF)",
            min_val=0.5, max_val=15.0, default_val=1.0, step=0.1, decimals=1, neutral_val=1.0, has_sub_bypass=False
        )
        self.row_iters = AstroParamRow(
            title="Iterations (Deconvolution)",
            min_val=1, max_val=30, default_val=10, step=1, decimals=0, neutral_val=0, has_sub_bypass=False
        )

        for row in [self.row_sigma, self.row_iters]:
            row.valueChanged.connect(lambda _: self.slider_changed.emit())
            row.bypassToggled.connect(lambda: self.bypass_toggled.emit())
            row.paramCommitted.connect(lambda: self.state_committed.emit())
            self.content_layout.addWidget(row)

        edge_hbox = QHBoxLayout()
        edge_hbox.setContentsMargins(0, 0, 0, 0)
        edge_hbox.setSpacing(6)
        edge_hbox.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.chk_anti_ringing = QCheckBox("Protect Edge (Anti-Ringing)")
        self.chk_anti_ringing.setChecked(False)
        self.chk_anti_ringing.setStyleSheet("""
            QCheckBox {
                color: #b0b0be;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                margin: 0px;
            }
            QCheckBox::indicator:checked {
                background-color: #00adb5;
                border: 1px solid #00fff2;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #23232c;
                border: 1px solid #454555;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked:disabled {
                background-color: #334155;
                border: 1px solid #475569;
                border-radius: 4px;
                image: url("assets/icons/check_ghost.png");
            }
            QCheckBox::indicator:unchecked:disabled {
                background-color: #1a1a22;
                border: 1px solid #334155;
                border-radius: 4px;
            }
        """)
        self.chk_anti_ringing.toggled.connect(lambda _: (self.bypass_toggled.emit(), self.state_committed.emit()))

        btn_help = HelpIcon("Protect Edge (Anti-Ringing)")
        edge_hbox.addWidget(self.chk_anti_ringing)
        edge_hbox.addWidget(btn_help)
        edge_hbox.addStretch()
        self.content_layout.addLayout(edge_hbox)
        self._update_header_style(False)

    def set_flat(self):
        self.row_sigma.set_value(1.0)
        self.row_iters.set_value(10)
        self.chk_anti_ringing.blockSignals(True)
        self.chk_anti_ringing.setChecked(False)
        self.chk_anti_ringing.blockSignals(False)
        self.set_dry_wet_ratio(100.0)

    def get_values(self):
        wet_val = float(self.get_dry_wet_ratio() * 100.0)
        return {
            'rl_sigma': self.row_sigma.get_actual_value(),
            'rl_iters': int(self.row_iters.get_actual_value()),
            'rl_anti_ringing': self.chk_anti_ringing.isChecked(),
            'rl_wet_mix': wet_val,
            'insert2_wet_mix': wet_val,
        }

    def set_values(self, d: dict):
        self.row_sigma.set_value(d.get('rl_sigma', 1.0))
        self.row_iters.set_value(d.get('rl_iters', 10))
        self.chk_anti_ringing.setChecked(bool(d.get('rl_anti_ringing', False)))
        self.set_dry_wet_ratio(float(d.get('rl_wet_mix', d.get('insert2_wet_mix', d.get('wet_mix', 100.0)))))


class ToneContrastInsert(CollapsibleInsertBox):
    def __init__(self, parent=None):
        super().__init__("INSERT 3: Tone & Contrast (Curves)", parent)

        self.row_black = AstroParamRow(
            title="Black Level (Sky Darkening)",
            min_val=0.000, max_val=0.250, default_val=0.000, step=0.005, decimals=3, neutral_val=0.000, has_sub_bypass=True
        )
        self.row_gamma = AstroParamRow(
            title="Midtones / Gamma",
            min_val=0.50, max_val=2.00, default_val=1.00, step=0.05, decimals=2, neutral_val=1.00, has_sub_bypass=True
        )
        self.row_contrast = AstroParamRow(
            title="Contrast (S-Curve)",
            min_val=1.00, max_val=2.50, default_val=1.00, step=0.05, decimals=2, neutral_val=1.00, has_sub_bypass=True
        )

        for row in [self.row_black, self.row_gamma, self.row_contrast]:
            row.valueChanged.connect(lambda _: self.slider_changed.emit())
            row.bypassToggled.connect(lambda: self.bypass_toggled.emit())
            row.paramCommitted.connect(lambda: self.state_committed.emit())
            self.content_layout.addWidget(row)
        self._update_header_style(False)

    def get_values(self):
        return {
            'tc_black': self.row_black.get_effective_value(),
            'tc_gamma': self.row_gamma.get_effective_value(),
            'tc_contrast': self.row_contrast.get_effective_value(),
            'tc_black_on': self.row_black.is_param_enabled(),
            'tc_gamma_on': self.row_gamma.is_param_enabled(),
            'tc_contrast_on': self.row_contrast.is_param_enabled(),
        }

    def set_values(self, d: dict):
        self.row_black.set_value(d.get('tc_black', 0.0))
        self.row_gamma.set_value(d.get('tc_gamma', 1.0))
        self.row_contrast.set_value(d.get('tc_contrast', 1.0))
        self.row_black.set_param_enabled(d.get('tc_black_on', False))
        self.row_gamma.set_param_enabled(d.get('tc_gamma_on', False))
        self.row_contrast.set_param_enabled(d.get('tc_contrast_on', False))


class ColorMonoInsert(CollapsibleInsertBox):
    def __init__(self, parent=None):
        super().__init__("INSERT 4: Color / Monochromatic Convert", parent)

        align_vbox = QVBoxLayout()
        align_vbox.setContentsMargins(0, 0, 0, 0)
        align_vbox.setSpacing(4)

        align_hbox = QHBoxLayout()
        align_hbox.setContentsMargins(0, 0, 0, 0)
        align_hbox.setSpacing(6)
        align_hbox.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.btn_rgb_align = QPushButton("Auto RGB Align")
        self.btn_rgb_align.setCheckable(True)
        self.btn_rgb_align.setFixedHeight(24)
        self.btn_rgb_align.setStyleSheet("""
            QPushButton {
                background-color: #242432;
                border: 1px solid #363646;
                color: #dcdce6;
                font-weight: bold;
                font-size: 10px;
                border-radius: 3px;
                padding: 0 8px;
            }
            QPushButton:hover {
                border-color: #00adb5;
            }
            QPushButton:checked {
                background-color: #00adb5;
                color: #111114;
                border-color: #00fff2;
            }
            QPushButton:disabled {
                background-color: #14141a;
                color: #555562;
                border-color: #22222a;
            }
            QPushButton:checked:disabled {
                background-color: #334155;
                color: #94a3b8;
                border-color: #475569;
            }
        """)
        self.btn_rgb_align.toggled.connect(lambda _: (self.bypass_toggled.emit(), self.state_committed.emit()))

        btn_help_align = HelpIcon("Auto RGB Align")
        align_hbox.addWidget(self.btn_rgb_align)
        align_hbox.addWidget(btn_help_align)
        align_hbox.addStretch()
        align_vbox.addLayout(align_hbox)

        self.lbl_shift_info = QLabel("Shift: R(0.0, 0.0) | B(0.0, 0.0) px")
        self.lbl_shift_info.setStyleSheet("color: #777785; font-size: 9px; padding-left: 0px;")
        align_vbox.addWidget(self.lbl_shift_info)
        self.content_layout.addLayout(align_vbox)

        h_box = QHBoxLayout()
        h_box.setContentsMargins(0, 0, 0, 0)
        h_box.setSpacing(6)
        h_box.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.chk_green = QCheckBox("Extract Green Channel (High-SNR)")
        self.chk_green.setChecked(False)
        self.chk_green.setStyleSheet("""
            QCheckBox {
                color: #b0b0be;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                margin: 0px;
            }
            QCheckBox::indicator:checked {
                background-color: #00adb5;
                border: 1px solid #00fff2;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #23232c;
                border: 1px solid #454555;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked:disabled {
                background-color: #334155;
                border: 1px solid #475569;
                border-radius: 4px;
                image: url("assets/icons/check_ghost.png");
            }
            QCheckBox::indicator:unchecked:disabled {
                background-color: #1a1a22;
                border: 1px solid #334155;
                border-radius: 4px;
            }
        """)
        self.chk_green.toggled.connect(self._on_green_toggled)

        btn_help_g = HelpIcon("Extract Green Channel")
        h_box.addWidget(self.chk_green)
        h_box.addWidget(btn_help_g)
        h_box.addStretch()
        self.content_layout.addLayout(h_box)
        self._update_header_style(False)

    def set_shifts_label(self, dx_r, dy_r, dx_b, dy_b):
        self.lbl_shift_info.setText(f"Shift: R({dx_r:+.2f}, {dy_r:+.2f}) | B({dx_b:+.2f}, {dy_b:+.2f}) px")

    def _on_green_toggled(self):
        self.bypass_toggled.emit()
        self.state_committed.emit()

    def get_values(self):
        return {
            'auto_rgb_align': self.btn_rgb_align.isChecked(),
            'extract_green': self.chk_green.isChecked()
        }

    def set_values(self, d: dict):
        self.btn_rgb_align.setChecked(bool(d.get('auto_rgb_align', False)))
        self.chk_green.setChecked(bool(d.get('extract_green', False)))


class LocalContrastInsert(CollapsibleInsertBox):
    def __init__(self, parent=None):
        super().__init__("INSERT 5: Local Contrast / Clarity", has_dry_wet=True, parent=parent)
        self.row_wet_mix = _DryWetRowAdapter(self.header.dry_wet_ctrl)

        mode_hbox = QHBoxLayout()
        mode_hbox.setContentsMargins(0, 0, 0, 0)
        mode_hbox.setSpacing(6)
        mode_hbox.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.lbl_mode = QLabel("Mode:")
        self.lbl_mode.setStyleSheet("color: #00adb5; font-size: 10px; font-weight: bold;")
        self.combo_mode = QComboBox()
        self.combo_mode.setFixedHeight(22)
        self.combo_mode.addItems(["CLAHE (Local Histogram)", "High-Pass (Macro Blur)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_help_mode = HelpIcon("Local Contrast Mode")
        mode_hbox.addWidget(self.lbl_mode)
        mode_hbox.addWidget(self.combo_mode)
        mode_hbox.addWidget(self.btn_help_mode)
        mode_hbox.addStretch()
        self.content_layout.addLayout(mode_hbox)

        self.row_amount = AstroParamRow(
            title="Clip Limit (Contrast)",
            min_val=1.00, max_val=5.00, default_val=2.00, step=0.10, decimals=2, neutral_val=1.00, has_sub_bypass=False
        )
        self.row_radius = AstroParamRow(
            title="Grid Size (Tile Area)",
            min_val=8, max_val=64, default_val=16, step=2, decimals=0, neutral_val=16, has_sub_bypass=False
        )

        for row in [self.row_amount, self.row_radius]:
            row.valueChanged.connect(lambda _: self.slider_changed.emit())
            row.bypassToggled.connect(lambda: self.bypass_toggled.emit())
            row.paramCommitted.connect(lambda: self.state_committed.emit())
            self.content_layout.addWidget(row)
        self._update_header_style(False)

    def _update_mode_controls_state(self, enabled: bool):
        primary = self._theme.get('primary', '#00adb5') if hasattr(self, '_theme') and self._theme else '#00adb5'
        if hasattr(self, 'lbl_mode') and self.lbl_mode is not None:
            self.lbl_mode.setStyleSheet(f"color: {primary if enabled else '#64748B'}; font-size: 10px; font-weight: bold;")
        if hasattr(self, 'combo_mode') and self.combo_mode is not None:
            self.combo_mode.setEnabled(enabled)
        if hasattr(self, 'btn_help_mode') and self.btn_help_mode is not None:
            self.btn_help_mode.setEnabled(enabled)

    def _configure_mode(self, is_hp: bool):
        if is_hp:
            self.row_amount.update_title_and_help("Clarity Amount")
            self.row_amount.set_range(min_val=0.00, max_val=3.00, step=0.05, default_val=0.80, neutral_val=0.00, decimals=2)
            self.row_radius.update_title_and_help("Sigma Radius (Blur)")
            self.row_radius.set_range(min_val=15.0, max_val=60.0, step=1.0, default_val=30.0, neutral_val=30.0, decimals=1)
        else:
            self.row_amount.update_title_and_help("Clip Limit (Contrast)")
            self.row_amount.set_range(min_val=1.00, max_val=5.00, step=0.10, default_val=2.00, neutral_val=1.00, decimals=2)
            self.row_radius.update_title_and_help("Grid Size (Tile Area)")
            self.row_radius.set_range(min_val=8, max_val=64, step=2, default_val=16, neutral_val=16, decimals=0)

    def _on_mode_changed(self, idx: int):
        is_hp = (idx == 1)
        self._configure_mode(is_hp)
        if is_hp:
            self.row_amount.set_value(0.80)
            self.row_radius.set_value(30.0)
        else:
            self.row_amount.set_value(2.00)
            self.row_radius.set_value(16.0)
        self.slider_changed.emit()
        self.state_committed.emit()

    def set_flat(self):
        self.combo_mode.blockSignals(True)
        self.combo_mode.setCurrentIndex(0)
        self._configure_mode(False)
        self.combo_mode.blockSignals(False)
        self.row_amount.set_value(1.00)
        self.row_radius.set_value(16.0)
        self.set_dry_wet_ratio(100.0)

    def get_values(self):
        is_hp = (self.combo_mode.currentIndex() == 1)
        wet_val = float(self.get_dry_wet_ratio() * 100.0)
        return {
            'lc_method': 'high_pass' if is_hp else 'clahe',
            'lc_amount': float(self.row_amount.get_actual_value()),
            'lc_radius': float(self.row_radius.get_actual_value()),
            'lc_wet_mix': wet_val,
            'insert5_wet_mix': wet_val,
        }

    def set_values(self, d: dict):
        method = str(d.get('lc_method', 'clahe')).lower()
        is_hp = method in ('high_pass', 'highpass', 'hp')
        self.combo_mode.blockSignals(True)
        self.combo_mode.setCurrentIndex(1 if is_hp else 0)
        self._configure_mode(is_hp)
        self.combo_mode.blockSignals(False)
        self.row_amount.set_value(float(d.get('lc_amount', 0.8 if is_hp else 2.0)))
        self.row_radius.set_value(float(d.get('lc_radius', 30.0 if is_hp else 16.0)))
        self.set_dry_wet_ratio(float(d.get('lc_wet_mix', d.get('insert5_wet_mix', d.get('wet_mix', 100.0)))))


class ColorEngineInsert(CollapsibleInsertBox):
    def __init__(self, parent=None):
        super().__init__("INSERT 6: Color Engine & Background", has_dry_wet=True, parent=parent)
        self.row_wet_mix = _DryWetRowAdapter(self.header.dry_wet_ctrl)
        self._bg_coords = None

        self.row_saturation = AstroParamRow(
            title="Color Saturation",
            min_val=0.00, max_val=3.00, default_val=1.00, step=0.05, decimals=2, neutral_val=1.00, has_sub_bypass=True
        )
        self.row_mineral = AstroParamRow(
            title="Mineral Moon Boost (Ti / Fe)",
            min_val=0.00, max_val=2.00, default_val=0.00, step=0.05, decimals=2, neutral_val=0.00, has_sub_bypass=True
        )

        for row in [self.row_saturation, self.row_mineral]:
            row.valueChanged.connect(lambda _: self.slider_changed.emit())
            row.bypassToggled.connect(lambda: self.bypass_toggled.emit())
            row.paramCommitted.connect(lambda: self.state_committed.emit())
            self.content_layout.addWidget(row)

        bg_vbox = QVBoxLayout()
        bg_vbox.setContentsMargins(0, 0, 0, 0)
        bg_vbox.setSpacing(4)

        bg_hbox = QHBoxLayout()
        bg_hbox.setContentsMargins(0, 0, 0, 0)
        bg_hbox.setSpacing(6)
        bg_hbox.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.chk_bg_neutralize = QCheckBox("Neutralize Sky Background")
        self.chk_bg_neutralize.setChecked(False)
        self.chk_bg_neutralize.setStyleSheet("""
            QCheckBox {
                color: #b0b0be;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                margin: 0px;
            }
            QCheckBox::indicator:checked {
                background-color: #00adb5;
                border: 1px solid #00fff2;
                border-radius: 4px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #23232c;
                border: 1px solid #454555;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked:disabled {
                background-color: #334155;
                border: 1px solid #475569;
                border-radius: 4px;
                image: url("assets/icons/check_ghost.png");
            }
            QCheckBox::indicator:unchecked:disabled {
                background-color: #1a1a22;
                border: 1px solid #334155;
                border-radius: 4px;
            }
        """)
        self.chk_bg_neutralize.toggled.connect(self._on_bg_toggled)

        btn_help_bg = HelpIcon("Neutralize Sky Background")
        bg_hbox.addWidget(self.chk_bg_neutralize)
        bg_hbox.addWidget(btn_help_bg)
        bg_hbox.addStretch()
        bg_vbox.addLayout(bg_hbox)

        self.lbl_bg_info = QLabel("Target: Auto-sample dark sky")
        self.lbl_bg_info.setStyleSheet("color: #777785; font-size: 9px; padding-left: 0px;")
        bg_vbox.addWidget(self.lbl_bg_info)
        self.content_layout.addLayout(bg_vbox)
        self._update_header_style(False)

    def _on_bg_toggled(self):
        self.bypass_toggled.emit()
        self.state_committed.emit()

    def set_bg_coords(self, coords):
        self._bg_coords = coords
        if coords is not None:
            x, y, w, h = coords
            self.lbl_bg_info.setText(f"Target ROI: ({int(x)}, {int(y)}) [{int(w)}x{int(h)}]")
        else:
            self.lbl_bg_info.setText("Target: Auto-sample dark sky")

    def set_flat(self):
        self.row_saturation.set_value(1.00)
        self.row_saturation.set_param_enabled(False)
        self.row_mineral.set_value(0.00)
        self.row_mineral.set_param_enabled(False)
        self.chk_bg_neutralize.blockSignals(True)
        self.chk_bg_neutralize.setChecked(False)
        self.chk_bg_neutralize.blockSignals(False)
        self.set_bg_coords(None)
        self.set_dry_wet_ratio(100.0)

    def get_values(self):
        wet_val = float(self.get_dry_wet_ratio() * 100.0)
        return {
            'cs_saturation': float(self.row_saturation.get_effective_value()),
            'cs_mineral': float(self.row_mineral.get_effective_value()),
            'cs_saturation_on': self.row_saturation.is_param_enabled(),
            'cs_mineral_on': self.row_mineral.is_param_enabled(),
            'cs_bg_neutralize': self.chk_bg_neutralize.isChecked(),
            'cs_bg_coords': self._bg_coords,
            'cs_wet_mix': wet_val,
            'insert6_wet_mix': wet_val,
        }

    def set_values(self, d: dict):
        self.row_saturation.set_value(float(d.get('cs_saturation', 1.0)))
        self.row_mineral.set_value(float(d.get('cs_mineral', 0.0)))

        if 'cs_saturation_on' in d:
            self.row_saturation.set_param_enabled(bool(d['cs_saturation_on']))
        elif float(d.get('cs_saturation', 1.0)) != 1.0:
            self.row_saturation.set_param_enabled(True)
        else:
            self.row_saturation.set_param_enabled(False)

        if 'cs_mineral_on' in d:
            self.row_mineral.set_param_enabled(bool(d['cs_mineral_on']))
        elif float(d.get('cs_mineral', 0.0)) > 0.0:
            self.row_mineral.set_param_enabled(True)
        else:
            self.row_mineral.set_param_enabled(False)

        self.chk_bg_neutralize.blockSignals(True)
        self.chk_bg_neutralize.setChecked(bool(d.get('cs_bg_neutralize', False)))
        self.chk_bg_neutralize.blockSignals(False)
        self.set_bg_coords(d.get('cs_bg_coords', None))
        self.set_dry_wet_ratio(float(d.get('cs_wet_mix', d.get('insert6_wet_mix', d.get('wet_mix', 100.0)))))





class PsfProfileWidget(QWidget):
    """Микро-виджет для отображения кривой Гауссианы и измеренных точек сечения PSF."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 95)
        self.setFixedHeight(95)
        self._data = None

    def set_psf_data(self, data: dict):
        self._data = data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#121218"))
        painter.setPen(QPen(QColor("#2c2c3a"), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        if not self._data:
            painter.setPen(QColor("#555566"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No PSF Data")
            return

        xs = self._data.get('xs', [])
        data_cut = self._data.get('data_cut', [])
        model_cut = self._data.get('model_cut', [])
        if not xs or not model_cut:
            return

        bg = self._data.get('background', 0.0)
        peak = self._data.get('peak', 1.0)
        fwhm = self._data.get('fwhm', 3.0)

        pad_l, pad_r = 28, 12
        pad_t, pad_b = 10, 16
        cw = max(10, w - pad_l - pad_r)
        ch = max(10, h - pad_t - pad_b)

        y_min = max(0.0, bg - 0.05)
        y_max = max(y_min + 0.1, peak + 0.05)
        x_min = min(xs)
        x_max = max(xs)
        x_span = max(1e-4, x_max - x_min)
        y_span = max(1e-4, y_max - y_min)

        def to_screen(x_val, y_val):
            sx = pad_l + (x_val - x_min) / x_span * cw
            sy = pad_t + ch - (y_val - y_min) / y_span * ch
            return QPointF(sx, sy)

        # Сетка уровня фона (Background)
        painter.setPen(QPen(QColor(60, 60, 80, 140), 1, Qt.PenStyle.DashLine))
        p_bg_l = to_screen(x_min, bg)
        p_bg_r = to_screen(x_max, bg)
        painter.drawLine(p_bg_l, p_bg_r)

        # Сетка полувысоты (Half-Maximum)
        half_max = bg + (peak - bg) * 0.5
        p_hm_l = to_screen(x_min, half_max)
        p_hm_r = to_screen(x_max, half_max)
        painter.setPen(QPen(QColor(229, 169, 59, 120), 1, Qt.PenStyle.DotLine))
        painter.drawLine(p_hm_l, p_hm_r)

        # Планка FWHM на уровне полувысоты
        half_fwhm = fwhm / 2.0
        fwhm_p1 = to_screen(-half_fwhm, half_max)
        fwhm_p2 = to_screen(half_fwhm, half_max)
        painter.setPen(QPen(QColor("#e5a93b"), 2))
        painter.drawLine(fwhm_p1, fwhm_p2)
        painter.drawLine(QPointF(fwhm_p1.x(), fwhm_p1.y() - 3), QPointF(fwhm_p1.x(), fwhm_p1.y() + 3))
        painter.drawLine(QPointF(fwhm_p2.x(), fwhm_p2.y() - 3), QPointF(fwhm_p2.x(), fwhm_p2.y() + 3))

        # Точки реального сечения
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(130, 130, 160, 160)))
        for x_val, y_val in zip(xs, data_cut):
            pt = to_screen(x_val, y_val)
            painter.drawEllipse(pt, 1.8, 1.8)

        # Аппроксимационная кривая Гауссианы
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#00fff2"), 2))
        curve_path = QPainterPath()
        first = True
        for x_val, y_val in zip(xs, model_cut):
            pt = to_screen(x_val, y_val)
            if first:
                curve_path.moveTo(pt)
                first = False
            else:
                curve_path.lineTo(pt)
        painter.drawPath(curve_path)

        # Подписи числовых уровней
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#777788"))
        painter.drawText(QRectF(2, p_bg_l.y() - 6, pad_l - 4, 12), Qt.AlignmentFlag.AlignRight, f"{bg:.2f}")
        painter.drawText(QRectF(2, to_screen(x_min, peak).y() - 6, pad_l - 4, 12), Qt.AlignmentFlag.AlignRight, f"{peak:.2f}")
        painter.drawText(QRectF(pad_l, h - pad_b, cw, pad_b), Qt.AlignmentFlag.AlignCenter, "Radius (px)")


ASTRO_HELP_TEXTS.update({
    "PSF FWHM": (
        "<b>FWHM — Full Width at Half Maximum</b><br><br>"
        "Ширина профиля звезды (или мелкой детали) на половине высоты пика, в пикселях. "
        "Это главная мера резкости кадра: на неё влияют сиинг атмосферы, фокусировка и дифракция оптики. "
        "Чем меньше FWHM, тем резче изображение. Для Гауссианы FWHM ≈ 2.355·σ. "
        "Угловой размер: FWHM[″] = FWHM[px] · 206.265 · pixel[мкм] / F[мм]."
    ),
    "PSF Sigma": (
        "<b>σ — стандартное отклонение Гауссианы</b><br><br>"
        "Параметр ширины аппроксимирующей кривой Гаусса (в пикселях). Именно σ используется как радиус PSF "
        "при деконволюции Ричардсона-Люси: кнопка «Apply to Deconv PSF» подставляет его в Insert 2."
    ),
    "PSF Peak": (
        "<b>Peak — пиковая яркость</b><br><br>"
        "Яркость центра звезды (0…1 в нормированной шкале кадра), уже с учётом уровня фона. "
        "Если Peak близок к 1.0, звезда пересвечена (клиппинг) и FWHM будет занижен — выбирайте менее яркие звёзды. "
        "Слишком малый Peak (близкий к шуму) даёт ненадёжную аппроксимацию."
    ),
    "PSF Background": (
        "<b>Bg — уровень фона</b><br><br>"
        "Медианная яркость вокруг объекта (оценка по периметру окна замера): свечение неба и шум матрицы. "
        "Фон вычитается из профиля перед расчётом FWHM, поэтому у хорошего замера разница Peak − Bg велика "
        "по сравнению с шумом."
    ),
    "PSF Profile Chart": (
        "<b>Как читать график профиля</b><br><br>"
        "• <b>Серые точки</b> — реальная яркость пикселей вдоль радиуса от центра звезды.<br>"
        "• <b>Бирюзовая кривая</b> — аппроксимация Гауссианой (модель звёздного диска). "
        "Чем точнее точки ложатся на кривую, тем надёжнее замер.<br>"
        "• <b>Нижняя пунктирная линия</b> — уровень фона/шума (Bg).<br>"
        "• <b>Янтарная пунктирная линия и планка</b> — половина высоты пика; длина планки и есть FWHM.<br><br>"
        "Широкая «колоколообразная» кривая — размытая звезда; узкая и высокая — резкая. "
        "Сильный разброс точек вокруг кривой говорит о шуме, двойной звезде или неподходящей цели."
    ),
})


class PsfInspectorDialog(QDialog):
    """Компактное плавающее окно инспектора PSF / FWHM со своей заголовочной панелью."""
    applyPsfRequested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSF / FWHM Inspector")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setFixedWidth(300)
        self._current_sigma = 1.0
        self._drag_offset = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Заголовок: симметричные отступы слева/справа, крестик не липнет к рамке ---
        self.title_bar = QFrame()
        self.title_bar.setObjectName("PsfTitleBar")
        self.title_bar.setFixedHeight(28)
        tb = QHBoxLayout(self.title_bar)
        tb.setContentsMargins(12, 0, 12, 0)
        tb.setSpacing(6)
        self.lbl_title = QLabel("PSF / FWHM Inspector")
        self.lbl_title.setStyleSheet("color: #b0b0be; font-size: 11px; font-weight: bold;")
        self.btn_close = QPushButton("×")
        self.btn_close.setFixedSize(20, 20)
        self.btn_close.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_close.setToolTip("Закрыть инспектор (Esc)")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #ff5566;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                padding: 0 0 2px 0;
                margin: 0;
            }
            QPushButton:hover {
                background-color: #ff3344;
                color: #ffffff;
            }
        """)
        self.btn_close.clicked.connect(self.reject)
        tb.addWidget(self.lbl_title)
        tb.addStretch()
        tb.addWidget(self.btn_close)
        root.addWidget(self.title_bar)

        body = QVBoxLayout()
        body.setContentsMargins(12, 10, 12, 12)
        body.setSpacing(8)
        root.addLayout(body)

        metrics_box = QFrame()
        metrics_box.setObjectName("PsfMetrics")
        metrics_box.setStyleSheet(
            "QFrame#PsfMetrics { background-color: #1a1a24; border: 1px solid #2d2d3c; border-radius: 4px; }"
        )
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(8, 6, 8, 6)
        metrics_layout.setSpacing(4)

        def metric(label: QLabel, help_key: str) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(label)
            row.addWidget(HelpIcon(help_key))
            return row

        top_row = QHBoxLayout()
        self.lbl_fwhm = QLabel("FWHM: -- px")
        self.lbl_fwhm.setStyleSheet("color: #00fff2; font-size: 13px; font-weight: bold;")
        self.lbl_sigma = QLabel("σ: -- px")
        self.lbl_sigma.setStyleSheet("color: #e5a93b; font-size: 11px; font-weight: bold;")
        top_row.addLayout(metric(self.lbl_fwhm, "PSF FWHM"))
        top_row.addStretch()
        top_row.addLayout(metric(self.lbl_sigma, "PSF Sigma"))
        metrics_layout.addLayout(top_row)

        bot_row = QHBoxLayout()
        self.lbl_peak = QLabel("Peak: --")
        self.lbl_peak.setStyleSheet("color: #888898; font-size: 10px;")
        self.lbl_bg = QLabel("Bg: --")
        self.lbl_bg.setStyleSheet("color: #888898; font-size: 10px;")
        bot_row.addLayout(metric(self.lbl_peak, "PSF Peak"))
        bot_row.addStretch()
        bot_row.addLayout(metric(self.lbl_bg, "PSF Background"))
        metrics_layout.addLayout(bot_row)

        self.lbl_coord = QLabel("Pos: --")
        self.lbl_coord.setStyleSheet("color: #666675; font-size: 10px;")
        metrics_layout.addWidget(self.lbl_coord)
        body.addWidget(metrics_box)

        chart_head = QHBoxLayout()
        chart_head.setSpacing(4)
        lbl_chart = QLabel("Profile")
        lbl_chart.setStyleSheet("color: #8a8a99; font-size: 10px; font-weight: bold;")
        chart_head.addWidget(lbl_chart)
        chart_head.addWidget(HelpIcon("PSF Profile Chart"))
        chart_head.addStretch()
        body.addLayout(chart_head)

        self.chart = PsfProfileWidget(self)
        body.addWidget(self.chart)

        self.btn_apply = QPushButton("Apply to Deconv PSF")
        self.btn_apply.setFixedHeight(26)
        self.btn_apply.setToolTip("Передать рассчитанную сигму PSF в параметр Radius инсерта деконволюции и закрыть инспектор")
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #1e2632;
                color: #00fff2;
                border: 1px solid #00adb5;
                border-radius: 5px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00adb5;
                color: #111114;
            }
        """)
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        body.addWidget(self.btn_apply)

        self.setStyleSheet("""
            QDialog {
                background-color: #14141a;
                border: 1px solid #2e2e3e;
                border-radius: 8px;
            }
            QFrame#PsfTitleBar {
                background-color: #1a1a22;
                border-bottom: 1px solid #2a2a36;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
        """)

    # Перетаскивание окна за заголовок (рамки у окна нет)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= self.title_bar.height():
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def update_metrics(self, data: dict):
        if not data:
            return
        self._current_sigma = float(data.get('sigma', 1.0))
        fwhm = data.get('fwhm', 0.0)
        peak = data.get('peak', 0.0)
        bg = data.get('background', 0.0)
        cx, cy = data.get('center_scene', (0, 0))

        self.lbl_fwhm.setText(f"FWHM: {fwhm:.2f} px")
        self.lbl_sigma.setText(f"σ: {self._current_sigma:.2f} px")
        self.lbl_peak.setText(f"Peak: {peak:.3f}")
        self.lbl_bg.setText(f"Bg: {bg:.3f}")
        self.lbl_coord.setText(f"Pos: ({cx:.0f}, {cy:.0f})")
        self.btn_apply.setText(f"Apply to Deconv PSF (σ={self._current_sigma:.2f})")
        self.chart.set_psf_data(data)

    def _on_apply_clicked(self):
        """Передаёт σ в деконволюцию и закрывает инспектор (главное окно сбросит инструмент PSF)."""
        self.applyPsfRequested.emit(self._current_sigma)
        self.accept()


class TelescopeOpticsDialog(QDialog):
    """
    Диалог настройки оптических параметров телескопа и компенсации эллиптической
    орбиты Луны (расчёт расстояния до Земли по Меёсу / ELP-2000 или вручную).
    """

    params_preview = pyqtSignal(float, float, float)  # focal_len, pixel_size, moon_distance

    def __init__(
        self,
        focal_len: float,
        pixel_size: float,
        parent=None,
        moon_distance_km: float = DEFAULT_LUNAR_DISTANCE_KM,
        lunar_mode: str = "date",
        capture_date: Optional[Union[datetime.date, datetime.datetime]] = None,
        is_auto_detected: bool = False,
        on_preview_update: Optional[Callable[[float, float, float], None]] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Калибровка линейки и оптики телескопа / Telescope & Lunar Scale")
        self.setFixedWidth(380)

        self._preview_callback = on_preview_update
        self._is_auto_detected = bool(is_auto_detected)
        self._initial_focal_len = float(focal_len)
        self._initial_pixel_size = float(pixel_size)
        self._initial_moon_dist = float(moon_distance_km)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        lbl_desc = QLabel("Задайте параметры оптики телескопа и лунной дистанции для калибровки линейки:")
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        layout.addWidget(lbl_desc)

        # ---------------- 1. Блок параметров оптики ----------------
        box_optics = QFrame()
        box_optics.setObjectName("opticsCard")
        box_optics.setStyleSheet("QFrame#opticsCard { background-color: #1a1c24; border: 1px solid #2e384d; border-radius: 6px; padding: 10px; }")
        layout_optics = QVBoxLayout(box_optics)
        layout_optics.setContentsMargins(10, 10, 10, 10)
        layout_optics.setSpacing(8)

        lbl_opt_title = QLabel("🔭 Оптика телескопа (Telescope Optics)")
        lbl_opt_title.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 11px;")
        layout_optics.addWidget(lbl_opt_title)

        row1 = QHBoxLayout()
        lbl_f = QLabel("Фокусное расстояние (F, мм):")
        lbl_f.setStyleSheet("color: #dcdce6; font-size: 11px;")
        self.spin_f = QDoubleSpinBox()
        self.spin_f.setRange(20.0, 20000.0)
        self.spin_f.setSingleStep(50.0)
        self.spin_f.setValue(focal_len)
        self.spin_f.setFixedWidth(95)
        self.spin_f.setStyleSheet("background-color: #1e1e28; color: #00fff2; border: 1px solid #363648; border-radius: 4px;")
        row1.addWidget(lbl_f)
        row1.addStretch()
        row1.addWidget(self.spin_f)
        layout_optics.addLayout(row1)

        row2 = QHBoxLayout()
        lbl_p = QLabel("Размер пикселя матрицы (µm):")
        lbl_p.setStyleSheet("color: #dcdce6; font-size: 11px;")
        self.spin_p = QDoubleSpinBox()
        self.spin_p.setRange(0.5, 30.0)
        self.spin_p.setSingleStep(0.1)
        self.spin_p.setDecimals(2)
        self.spin_p.setValue(pixel_size)
        self.spin_p.setFixedWidth(95)
        self.spin_p.setStyleSheet("background-color: #1e1e28; color: #00fff2; border: 1px solid #363648; border-radius: 4px;")
        row2.addWidget(lbl_p)
        row2.addStretch()
        row2.addWidget(self.spin_p)
        layout_optics.addLayout(row2)

        self.lbl_scale = QLabel()
        self.lbl_scale.setStyleSheet("color: #e5a93b; font-weight: bold; font-size: 11px; margin-top: 2px;")
        self._update_scale_label()
        layout_optics.addWidget(self.lbl_scale)

        layout.addWidget(box_optics)

        # ---------------- 2. Блок лунной калибровки (Distance to Moon) ----------------
        box_lunar = QFrame()
        box_lunar.setObjectName("lunarCard")
        box_lunar.setStyleSheet("QFrame#lunarCard { background-color: #1a1c24; border: 1px solid #2e384d; border-radius: 6px; padding: 10px; }")
        layout_lunar = QVBoxLayout(box_lunar)
        layout_lunar.setContentsMargins(10, 10, 10, 10)
        layout_lunar.setSpacing(8)

        lbl_lunar_title = QLabel("🌙 Дистанция до Луны (Distance to Moon)")
        lbl_lunar_title.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 11px;")
        layout_lunar.addWidget(lbl_lunar_title)

        # Переключатель режима (QRadioButton)
        row_modes = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.radio_by_date = QRadioButton("По дате съёмки")
        self.radio_manual = QRadioButton("Вручную в км")
        self.mode_group.addButton(self.radio_by_date)
        self.mode_group.addButton(self.radio_manual)
        row_modes.addWidget(self.radio_by_date)
        row_modes.addWidget(self.radio_manual)
        row_modes.addStretch()
        layout_lunar.addLayout(row_modes)

        # Поле ввода даты съёмки + бейдж автоопределения EXIF
        row_date = QHBoxLayout()
        lbl_date = QLabel("Дата съёмки (UTC):")
        lbl_date.setStyleSheet("color: #dcdce6; font-size: 11px;")
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setFixedWidth(105)

        # Инициализация даты
        if capture_date is not None:
            if isinstance(capture_date, (datetime.datetime, datetime.date)):
                qd = QDate(capture_date.year, capture_date.month, capture_date.day)
            else:
                qd = QDate.currentDate()
        else:
            qd = QDate.currentDate()
        self.date_edit.setDate(qd)

        self.lbl_badge = QLabel("EXIF Auto-detected")
        self.lbl_badge.setStyleSheet(
            "background-color: #14532d; color: #4ade80; border: 1px solid #22c55e; "
            "border-radius: 4px; padding: 2px 6px; font-weight: bold; font-size: 9px;"
        )

        row_date.addWidget(lbl_date)
        row_date.addWidget(self.date_edit)
        row_date.addWidget(self.lbl_badge)
        row_date.addStretch()
        layout_lunar.addLayout(row_date)

        # Поле ввода / отображения расстояния
        row_dist = QHBoxLayout()
        lbl_dist = QLabel("Расстояние до Луны:")
        lbl_dist.setStyleSheet("color: #dcdce6; font-size: 11px;")
        self.distance_spinbox = QDoubleSpinBox()
        self.distance_spinbox.setRange(350000.0, 420000.0)
        self.distance_spinbox.setSingleStep(100.0)
        self.distance_spinbox.setDecimals(1)
        self.distance_spinbox.setSuffix(" km")
        self.distance_spinbox.setFixedWidth(120)
        self.distance_spinbox.setValue(moon_distance_km)
        row_dist.addWidget(lbl_dist)
        row_dist.addStretch()
        row_dist.addWidget(self.distance_spinbox)
        layout_lunar.addLayout(row_dist)

        self.lbl_lunar_hint = QLabel("Расчёт по теории Меёса (ELP-2000). Перигей ~363 300 км | Апогей ~405 500 км")
        self.lbl_lunar_hint.setWordWrap(True)
        self.lbl_lunar_hint.setStyleSheet("color: #718096; font-size: 10px; margin-top: 2px;")
        layout_lunar.addWidget(self.lbl_lunar_hint)

        layout.addWidget(box_lunar)

        # ---------------- 3. Кнопки сохранения и отмены ----------------
        btn_box = QHBoxLayout()
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setObjectName("btn_cancel")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = QPushButton("Сохранить")
        self.btn_ok.clicked.connect(self.accept)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_ok)
        layout.addLayout(btn_box)

        # Подключение сигналов
        self.spin_f.valueChanged.connect(self._on_optics_changed)
        self.spin_p.valueChanged.connect(self._on_optics_changed)
        self.radio_by_date.toggled.connect(self._on_mode_changed)
        self.date_edit.dateChanged.connect(self._on_date_changed)
        self.distance_spinbox.valueChanged.connect(self._on_distance_changed)

        # Установка исходного режима
        if lunar_mode == "manual":
            self.radio_manual.setChecked(True)
        else:
            self.radio_by_date.setChecked(True)
        self._update_ui_state()

        self.setStyleSheet("""
            QDialog {
                background-color: #16161e;
                border: 1px solid #2e2e3e;
                border-radius: 8px;
            }
            QLabel {
                background: transparent;
                border: none;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QDoubleSpinBox, QDateEdit {
                background-color: #1e1e28;
                color: #00fff2;
                border: 1px solid #363648;
                border-radius: 4px;
                padding: 3px 6px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
            }
            QDoubleSpinBox:focus, QDateEdit:focus {
                border: 1px solid #00fff2;
            }
            QDoubleSpinBox:disabled, QDateEdit:disabled {
                background-color: #12141a;
                color: #64748b;
                border: 1px solid #232733;
            }
            QRadioButton {
                color: #dcdce6;
                font-size: 11px;
                font-family: 'Segoe UI', Arial, sans-serif;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 1px solid #4a5568;
                background-color: #1a1c24;
            }
            QRadioButton::indicator:hover {
                border-color: #00fff2;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #00fff2;
                background-color: #00fff2;
            }
            QPushButton {
                background-color: #00adb5;
                color: #111114;
                font-weight: bold;
                font-size: 11px;
                padding: 5px 20px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #00fff2;
            }
            QPushButton:pressed {
                background-color: #00888f;
                color: #ffffff;
            }
            QPushButton#btn_cancel {
                background-color: #2e384d;
                color: #dcdce6;
            }
            QPushButton#btn_cancel:hover {
                background-color: #3b4764;
                color: #ffffff;
            }
            QCalendarWidget QWidget {
                alternate-background-color: #1a1c24;
                background-color: #16161e;
                color: #dcdce6;
            }
            QCalendarWidget QAbstractItemView:enabled {
                color: #dcdce6;
                background-color: #16161e;
                selection-background-color: #00adb5;
                selection-color: #111114;
            }
            QCalendarWidget QToolButton {
                color: #00fff2;
                background-color: #1a1c24;
            }
            QCalendarWidget QMenu {
                background-color: #1a1c24;
                color: #dcdce6;
            }
            QCalendarWidget QSpinBox {
                background-color: #1e1e28;
                color: #00fff2;
            }
        """)

    def _update_scale_label(self):
        f = max(1.0, self.spin_f.value())
        p = max(0.1, self.spin_p.value())
        scale = 206.265 * p / f
        self.lbl_scale.setText(f'Масштаб изображения: {scale:.2f} "/px')

    def _on_optics_changed(self):
        self._update_scale_label()
        self._notify_preview()

    def _on_mode_changed(self):
        self._update_ui_state()
        self._notify_preview()

    def _update_ui_state(self):
        by_date = self.radio_by_date.isChecked()
        if by_date:
            self.date_edit.setEnabled(True)
            self.distance_spinbox.setEnabled(False)
            if self._is_auto_detected:
                self.lbl_badge.show()
            else:
                self.lbl_badge.hide()
            # Пересчёт расстояния по текущей дате
            self._recalculate_distance_from_date()
        else:
            self.date_edit.setEnabled(False)
            self.distance_spinbox.setEnabled(True)
            self.lbl_badge.hide()

    def _recalculate_distance_from_date(self):
        qd = self.date_edit.date()
        py_d = datetime.date(qd.year(), qd.month(), qd.day())
        dist = calculate_lunar_distance(py_d, round_digits=1)
        self.distance_spinbox.blockSignals(True)
        self.distance_spinbox.setValue(dist)
        self.distance_spinbox.blockSignals(False)

    def _on_date_changed(self):
        if self.radio_by_date.isChecked():
            self._recalculate_distance_from_date()
            self._notify_preview()

    def _on_distance_changed(self):
        self._notify_preview()

    def _notify_preview(self):
        f = self.spin_f.value()
        p = self.spin_p.value()
        dist = self.distance_spinbox.value()
        self.params_preview.emit(f, p, dist)
        if self._preview_callback is not None:
            try:
                self._preview_callback(f, p, dist)
            except Exception:
                pass

    def get_focal_len(self) -> float:
        return float(self.spin_f.value())

    def get_pixel_size(self) -> float:
        return float(self.spin_p.value())

    def get_moon_distance(self) -> float:
        return float(self.distance_spinbox.value())

    def get_lunar_mode(self) -> str:
        return "date" if self.radio_by_date.isChecked() else "manual"

    def get_capture_date(self) -> datetime.date:
        qd = self.date_edit.date()
        return datetime.date(qd.year(), qd.month(), qd.day())

    def is_auto_detected(self) -> bool:
        return self._is_auto_detected and self.radio_by_date.isChecked()

    def get_values(self):
        """Возвращает кортеж (f_mm, pixel_um, lunar_distance_km)."""
        return self.spin_f.value(), self.spin_p.value(), self.distance_spinbox.value()


class PhotoshopToolButton(QToolButton):
    """Кнопка инструмента в стиле Photoshop с маркером выпадающего меню (flyout arrow)."""
    flyout_requested = pyqtSignal(QPoint)

    def __init__(self, name: str, icon_text: str, tooltip: str, has_flyout: bool = False, parent=None):
        super().__init__(parent)
        self.tool_name = name
        self.has_flyout = has_flyout
        self.setText(icon_text)
        self.setToolTip(tooltip)
        self.setCheckable(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(36, 36)
        self.setMouseTracking(True)
        self._press_in_corner = False
        self._press_pos = None

        self._long_press_timer = QTimer(self)
        self._long_press_timer.setSingleShot(True)
        self._long_press_timer.setInterval(320)
        self._long_press_timer.timeout.connect(self._on_long_press)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.has_flyout:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            w = float(self.width())
            h = float(self.height())
            poly = QPolygonF([
                QPointF(w - 3.0, h - 3.0),
                QPointF(w - 8.5, h - 3.0),
                QPointF(w - 3.0, h - 8.5)
            ])
            painter.setPen(Qt.PenStyle.NoPen)
            if self.isChecked():
                painter.setBrush(QBrush(QColor("#00fff2")))
            elif self.underMouse():
                painter.setBrush(QBrush(QColor("#dcdce6")))
            else:
                painter.setBrush(QBrush(QColor("#777788")))
            painter.drawPolygon(poly)

    def _is_in_corner(self, pos: QPoint) -> bool:
        return pos.x() >= self.width() - 12 and pos.y() >= self.height() - 12

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.flyout_requested.emit(event.globalPosition().toPoint())
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.globalPosition().toPoint()
            if self.has_flyout and self._is_in_corner(event.pos()):
                self._press_in_corner = True
                self.flyout_requested.emit(self._press_pos)
                event.accept()
                return
            self._press_in_corner = False
            if self.has_flyout:
                self._long_press_timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._long_press_timer.isActive():
            self._long_press_timer.stop()
        if self._press_in_corner:
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _on_long_press(self):
        if self._press_pos is not None:
            self.flyout_requested.emit(self._press_pos)


class ToolboxPanel(QFrame):
    """Узкая левая панель инструментов в стиле Photoshop: режимы и геометрия кадра."""
    tool_changed = pyqtSignal(str)
    rotate_cw = pyqtSignal()
    rotate_ccw = pyqtSignal()
    flip_h = pyqtSignal()
    flip_v = pyqtSignal()
    apply_crop = pyqtSignal()
    loupe_zoom_changed = pyqtSignal(float)
    ruler_unit_changed = pyqtSignal(str)
    ruler_optics_changed = pyqtSignal(float, float)
    ruler_optics_requested = pyqtSignal()
    ruler_submode_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toolboxPanel")
        self.setFixedWidth(48)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("""
            QFrame#toolboxPanel {
                background-color: #14141a;
                border: 1px solid #282834;
                border-radius: 6px;
            }
            QToolButton {
                background-color: #1c1c24;
                border: 1px solid #2c2c38;
                border-radius: 5px;
                color: #c8c8d4;
                font-size: 18px;
                font-weight: normal;
                padding: 0;
            }
            QToolButton:hover {
                background-color: #262632;
                border-color: #00adb5;
                color: #ffffff;
            }
            QToolButton:checked {
                background-color: #14383c;
                border-color: #00fff2;
                color: #00fff2;
            }
            QToolButton:pressed {
                background-color: #00adb5;
                color: #111114;
            }
            QPushButton#toolboxApply {
                background-color: #00adb5;
                color: #111114;
                font-weight: bold;
                font-size: 12px;
                border: none;
                border-radius: 4px;
                padding: 0;
            }
            QPushButton#toolboxApply:hover {
                background-color: #00fff2;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._mode_buttons = {}
        self._loupe_zoom = 2.0
        self._ruler_focal_len = 1000.0
        self._ruler_pixel_size = 3.76
        self._ruler_lunar_dist = DEFAULT_LUNAR_DISTANCE_KM
        self._ruler_unit = "all"
        self._ruler_submode = "ruler"
        self._theme = THEMES['A']

        # 1. Loupe Button with Flyout
        self.btn_loupe = self._make_mode_button(
            "loupe", "⌕", "Инспектор лупы (Z / Alt)\n[ПКМ / стрелка: выбор зума]", has_flyout=True
        )
        self.btn_loupe.flyout_requested.connect(self._show_loupe_menu)
        layout.addWidget(self.btn_loupe)

        # 2. Crop Button (Icon ⌗)
        self.btn_crop = self._make_mode_button(
            "crop", "⌗", "Кадрирование / Crop Tool (C) — Enter: применить, Esc: отмена"
        )
        layout.addWidget(self.btn_crop)

        self.btn_apply_crop = QPushButton("✓")
        self.btn_apply_crop.setObjectName("toolboxApply")
        self.btn_apply_crop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_apply_crop.setFixedSize(36, 22)
        self.btn_apply_crop.setVisible(False)
        self.btn_apply_crop.setToolTip("Применить кадрирование (Enter)")
        self.btn_apply_crop.clicked.connect(self.apply_crop.emit)
        layout.addWidget(self.btn_apply_crop)

        # 3. PSF / FWHM Inspector Button (Icon ⯎, Key P)
        self.btn_psf = self._make_mode_button(
            "psf", "⯎", "Инспектор PSF / FWHM (P) — кликните на звезду/кратер"
        )
        layout.addWidget(self.btn_psf)

        # 4. Pixel & Angular Ruler Button with Flyout (Icon ⤢, Key M)
        self.btn_ruler = self._make_mode_button(
            "ruler", "⤢", "Линейка (M) — зажмите ЛКМ и протяните\n[ПКМ / стрелка: оптика и единицы]", has_flyout=True
        )
        self.btn_ruler.flyout_requested.connect(self._show_ruler_menu)
        layout.addWidget(self.btn_ruler)

        # 5. Profile Button (Key L)
        self.btn_profile = self._make_mode_button(
            "profile", "∿", "Инспектор среза яркости / Profile Tool (L)"
        )
        layout.addWidget(self.btn_profile)

        # Separator between Tools and Rotate
        sep1 = QFrame()
        sep1.setFixedHeight(1)
        sep1.setFixedWidth(36)
        sep1.setStyleSheet("background-color: #2e2e3e; margin: 3px 0;")
        layout.addWidget(sep1)

        # 6. Rotate Buttons
        self.btn_rot_cw = self._make_action_button(
            "⟳", "Поворот на 90° по часовой стрелке (R)"
        )
        self.btn_rot_ccw = self._make_action_button(
            "⟲", "Поворот на 90° против часовой стрелки (Shift+R)"
        )
        layout.addWidget(self.btn_rot_cw)
        layout.addWidget(self.btn_rot_ccw)

        # Separator between Rotate and Flip
        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setFixedWidth(36)
        sep2.setStyleSheet("background-color: #2e2e3e; margin: 3px 0;")
        layout.addWidget(sep2)

        # 7. Flip Buttons
        self.btn_flip_h = self._make_action_button(
            "⇄", "Зеркальное отражение по горизонтали (H)"
        )
        self.btn_flip_v = self._make_action_button(
            "⇅", "Зеркальное отражение по вертикали (V)"
        )
        layout.addWidget(self.btn_flip_h)
        layout.addWidget(self.btn_flip_v)

        self.btn_rot_cw.clicked.connect(self.rotate_cw.emit)
        self.btn_rot_ccw.clicked.connect(self.rotate_ccw.emit)
        self.btn_flip_h.clicked.connect(self.flip_h.emit)
        self.btn_flip_v.clicked.connect(self.flip_v.emit)

        layout.addStretch()

    def apply_theme(self, theme: dict):
        """Динамическое применение цветовой темы (State A/B) к панели инструментов."""
        self._theme = theme
        bright = theme.get('bright', '#00fff2')
        primary = theme.get('primary', '#00adb5')
        bg_accent = theme.get('bg_accent', '#14383c')
        self.setStyleSheet(f"""
            QFrame#toolboxPanel {{
                background-color: #14141a;
                border: 1px solid #282834;
                border-radius: 6px;
            }}
            QToolButton {{
                background-color: #1c1c24;
                border: 1px solid #2c2c38;
                border-radius: 5px;
                color: #c8c8d4;
                font-size: 18px;
                font-weight: normal;
                padding: 0;
            }}
            QToolButton:hover {{
                background-color: #262632;
                border-color: {bright};
                color: #ffffff;
            }}
            QToolButton:checked {{
                background-color: {bg_accent};
                border-color: {bright};
                color: {bright};
            }}
            QToolButton:pressed {{
                background-color: {primary};
                color: #111114;
            }}
            QPushButton#toolboxApply {{
                background-color: {primary};
                color: #111114;
                font-weight: bold;
                font-size: 12px;
                border: none;
                border-radius: 4px;
                padding: 0;
            }}
            QPushButton#toolboxApply:hover {{
                background-color: {bright};
            }}
        """)

    def _menu_qss(self) -> str:
        return """
            QMenu {
                background-color: #1c1c24;
                border: 1px solid #363648;
                border-radius: 4px;
                color: #dcdce6;
                padding: 4px;
            }
            QMenu::item {
                padding: 4px 18px 4px 20px;
                font-size: 11px;
            }
            QMenu::item:selected {
                background-color: #00adb5;
                color: #111114;
                border-radius: 3px;
            }
            QMenu::separator {
                height: 1px;
                background-color: #2e2e3e;
                margin: 4px 6px;
            }
        """

    def _show_loupe_menu(self, pos: QPoint):
        menu = QMenu(self)
        menu.setStyleSheet(self._menu_qss())
        presets = [("2x (200%)", 2.0), ("4x (400%)", 4.0), ("8x (800%)", 8.0), ("16x (1600%)", 16.0)]
        for label, factor in presets:
            act = menu.addAction(label)
            act.setCheckable(True)
            if abs(self._loupe_zoom - factor) < 1e-3:
                act.setChecked(True)
            act.triggered.connect(lambda ch, f=factor: self._set_loupe_zoom(f))
        menu.exec(pos)

    def _set_loupe_zoom(self, factor: float):
        self._loupe_zoom = factor
        self.loupe_zoom_changed.emit(factor)

    def _show_ruler_menu(self, pos: QPoint):
        menu = QMenu(self)
        menu.setStyleSheet(self._menu_qss())

        # Режим измерений (Measurement Mode)
        mode_menu = menu.addMenu("📐 Режим измерений (Mode)")
        mode_menu.setStyleSheet(self._menu_qss())
        meas_modes = [
            ("📏 Прямая линейка (Linear Ruler)", "ruler"),
            ("⭕ Привязка диска Луны (Moon Disk Alignment)", "disk"),
            ("⬭ Эллипс кратера (Crater Ellipse)", "ellipse"),
        ]
        for label, m_key in meas_modes:
            act = mode_menu.addAction(label)
            act.setCheckable(True)
            if getattr(self, '_ruler_submode', 'ruler') == m_key:
                act.setChecked(True)
            act.triggered.connect(lambda ch, m=m_key: self._set_ruler_submode(m))

        menu.addSeparator()

        # Единицы измерения
        unit_menu = menu.addMenu("📏 Единицы измерения (Units)")
        unit_menu.setStyleSheet(self._menu_qss())
        units = [
            ("Пиксели (px)", "px"),
            ("Угловые секунды (arcsec)", "arcsec"),
            ("Километры (km, Луна)", "km"),
            ("Все (All: px + arcsec)", "all")
        ]
        for label, u in units:
            act = unit_menu.addAction(label)
            act.setCheckable(True)
            if self._ruler_unit == u:
                act.setChecked(True)
            act.triggered.connect(lambda ch, unit_val=u: self._set_ruler_unit(unit_val))

        menu.addSeparator()

        # Фокусное расстояние
        focal_menu = menu.addMenu("🔭 Фокусное расстояние (Focal)")
        focal_menu.setStyleSheet(self._menu_qss())
        focal_presets = [
            ("750 мм (150/750 Newt)", 750.0),
            ("1000 мм (200/1000 Newt)", 1000.0),
            ("1200 мм (250/1200 Dob)", 1200.0),
            ("1500 мм (Celestron/Mak)", 1500.0),
        ]
        for label, f_val in focal_presets:
            act = focal_menu.addAction(label)
            act.setCheckable(True)
            if abs(self._ruler_focal_len - f_val) < 1.0:
                act.setChecked(True)
            act.triggered.connect(lambda ch, f=f_val: self._set_focal_len(f))
        focal_menu.addSeparator()
        act_custom_f = focal_menu.addAction("Пользовательское...")
        act_custom_f.triggered.connect(self.ruler_optics_requested.emit)

        # Размер пикселя
        pixel_menu = menu.addMenu("📷 Размер пикселя (Pixel Pitch)")
        pixel_menu.setStyleSheet(self._menu_qss())
        pixel_presets = [
            ("2.90 µm (IMX585 / IMX462)", 2.90),
            ("3.76 µm (IMX571 / IMX533)", 3.76),
        ]
        for label, p_val in pixel_presets:
            act = pixel_menu.addAction(label)
            act.setCheckable(True)
            if abs(self._ruler_pixel_size - p_val) < 0.05:
                act.setChecked(True)
            act.triggered.connect(lambda ch, p=p_val: self._set_pixel_size(p))
        pixel_menu.addSeparator()
        act_custom_p = pixel_menu.addAction("Пользовательское...")
        act_custom_p.triggered.connect(self.ruler_optics_requested.emit)

        menu.addSeparator()
        act_moon = menu.addAction(f"🌙 Расстояние до Луны ({self._ruler_lunar_dist:.0f} км)...")
        act_moon.triggered.connect(self.ruler_optics_requested.emit)
        act_opt = menu.addAction("⚙ Настройки оптики и калибровки...")
        act_opt.triggered.connect(self.ruler_optics_requested.emit)

        menu.exec(pos)

    def set_lunar_distance(self, dist_km: float):
        self._ruler_lunar_dist = float(dist_km)

    def set_ruler_submode(self, mode: str):
        self._ruler_submode = mode

    def _set_ruler_submode(self, mode: str):
        self._ruler_submode = mode
        self.ruler_submode_changed.emit(mode)

    def set_optics(self, f: float, p: float):
        self._ruler_focal_len = float(f)
        self._ruler_pixel_size = float(p)

    def _set_ruler_unit(self, unit: str):
        self._ruler_unit = unit
        self.ruler_unit_changed.emit(unit)

    def _set_focal_len(self, f: float):
        self._ruler_focal_len = f
        self.ruler_optics_changed.emit(self._ruler_focal_len, self._ruler_pixel_size)

    def _set_pixel_size(self, p: float):
        self._ruler_pixel_size = p
        self.ruler_optics_changed.emit(self._ruler_focal_len, self._ruler_pixel_size)

    def toggle_tool(self, name: str):
        if name in self._mode_buttons:
            btn = self._mode_buttons[name]
            btn.setChecked(not btn.isChecked())
            self._on_mode_clicked(name, btn.isChecked())

    def _make_mode_button(self, name: str, icon_text: str, tooltip: str, has_flyout: bool = False) -> PhotoshopToolButton:
        btn = PhotoshopToolButton(name, icon_text, tooltip, has_flyout=has_flyout)
        btn.clicked.connect(lambda checked, n=name: self._on_mode_clicked(n, checked))
        self._mode_buttons[name] = btn
        return btn

    def _make_action_button(self, icon_text: str, tooltip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(icon_text)
        btn.setToolTip(tooltip)
        btn.setCheckable(False)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setFixedSize(36, 36)
        return btn

    def _on_mode_clicked(self, name: str, checked: bool):
        if checked:
            for other, btn in self._mode_buttons.items():
                if other != name:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
            self.btn_apply_crop.setVisible(name == "crop")
            self.tool_changed.emit(name)
        else:
            self.btn_apply_crop.setVisible(False)
            self.tool_changed.emit("")

    def clear_tool(self):
        for btn in self._mode_buttons.values():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
        self.btn_apply_crop.setVisible(False)
        self.tool_changed.emit("")


# =============================================================================
# ПРОФЕССИОНАЛЬНЫЙ СПЕКТРО-ЯРКОСТНЫЙ АНАЛИЗАТОР (HISTOGRAM BLOCK)
# =============================================================================

class AstroHistogramHeader(QFrame):
    """Шапка блока гистограммы с переключателями [FULL|AUTO], [LOG|LIN], [LUM|RGB] и кнопкой справки [?]."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hist_widget = None
        self._active_state = 'A'
        self._accent = '#00E5FF'
        self._bright = '#00fff2'

        self.setFixedHeight(22)
        self.setStyleSheet("background: transparent; border: none;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.lbl_title = ElidedLabel("LUMINANCE HISTOGRAM", self)
        self.lbl_title.setStyleSheet("color: #888899; font-weight: bold; font-size: 11px; letter-spacing: 0.5px;")
        self.lbl_title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.lbl_title)
        layout.addStretch()

        # Группа режима выборки: [ FULL | AUTO ]
        self.roi_box = QFrame()
        self.roi_box.setFixedHeight(20)
        self.roi_box.setStyleSheet("background: #10131A; border: 1px solid #1E293B; border-radius: 4px; padding: 0px;")
        roi_hbox = QHBoxLayout(self.roi_box)
        roi_hbox.setContentsMargins(1, 1, 1, 1)
        roi_hbox.setSpacing(1)

        self.btn_full = QPushButton("FULL")
        self.btn_auto = QPushButton("AUTO")
        self.btn_full.setCheckable(True)
        self.btn_auto.setCheckable(True)
        self.btn_full.setChecked(True)
        self.btn_full.setFixedSize(26, 18)
        self.btn_auto.setFixedSize(26, 18)
        self.btn_full.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_auto.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_full.setToolTip("Полный кадр: гистограмма строится по всему снимку целиком")
        self.btn_auto.setToolTip("Auto-Mask ROI: автоматическое отсечение чёрного космоса и расчёт по диску планеты")

        self.roi_group = QButtonGroup(self)
        self.roi_group.setExclusive(True)
        self.roi_group.addButton(self.btn_full)
        self.roi_group.addButton(self.btn_auto)

        roi_hbox.addWidget(self.btn_full)
        roi_hbox.addWidget(self.btn_auto)
        layout.addWidget(self.roi_box)

        # Группа переключения шкалы: [ LOG | LIN ]
        self.scale_box = QFrame()
        self.scale_box.setFixedHeight(20)
        self.scale_box.setStyleSheet("background: #10131A; border: 1px solid #1E293B; border-radius: 4px; padding: 0px;")
        scale_hbox = QHBoxLayout(self.scale_box)
        scale_hbox.setContentsMargins(1, 1, 1, 1)
        scale_hbox.setSpacing(1)

        self.btn_log = QPushButton("LOG")
        self.btn_lin = QPushButton("LIN")
        self.btn_log.setCheckable(True)
        self.btn_lin.setCheckable(True)
        self.btn_log.setChecked(True)
        self.btn_log.setFixedSize(24, 18)
        self.btn_lin.setFixedSize(24, 18)
        self.btn_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_lin.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_log.setToolTip("Логарифмическая шкала по Y (раскрывает полутона космоса)")
        self.btn_lin.setToolTip("Линейная шкала по Y (стандартное распределение)")

        self.scale_group = QButtonGroup(self)
        self.scale_group.setExclusive(True)
        self.scale_group.addButton(self.btn_log)
        self.scale_group.addButton(self.btn_lin)

        scale_hbox.addWidget(self.btn_log)
        scale_hbox.addWidget(self.btn_lin)
        layout.addWidget(self.scale_box)

        # Группа переключения каналов: [ LUM | RGB ]
        self.chan_box = QFrame()
        self.chan_box.setFixedHeight(20)
        self.chan_box.setStyleSheet("background: #10131A; border: 1px solid #1E293B; border-radius: 4px; padding: 0px;")
        chan_hbox = QHBoxLayout(self.chan_box)
        chan_hbox.setContentsMargins(1, 1, 1, 1)
        chan_hbox.setSpacing(1)

        self.btn_lum = QPushButton("LUM")
        self.btn_rgb = QPushButton("RGB")
        self.btn_lum.setCheckable(True)
        self.btn_rgb.setCheckable(True)
        self.btn_lum.setChecked(True)
        self.btn_lum.setFixedSize(24, 18)
        self.btn_rgb.setFixedSize(24, 18)
        self.btn_lum.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rgb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_lum.setToolTip("Режим Luminance: интегральная яркость Rec.709")
        self.btn_rgb.setToolTip("Мультиканальный режим: спектры Red, Green и Blue с наложением")

        self.chan_group = QButtonGroup(self)
        self.chan_group.setExclusive(True)
        self.chan_group.addButton(self.btn_lum)
        self.chan_group.addButton(self.btn_rgb)

        chan_hbox.addWidget(self.btn_lum)
        chan_hbox.addWidget(self.btn_rgb)
        layout.addWidget(self.chan_box)

        # Кнопка контекстной справки: [ ? ]
        self.btn_help = QPushButton("?")
        self.btn_help.setFixedSize(16, 18)
        self.btn_help.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_help.setToolTip("Справка по профессиональному чтению гистограммы")
        self.btn_help.clicked.connect(self._open_help_dialog)
        layout.addWidget(self.btn_help)

        self.btn_full.clicked.connect(self._on_roi_changed)
        self.btn_auto.clicked.connect(self._on_roi_changed)
        self.btn_log.clicked.connect(self._on_scale_changed)
        self.btn_lin.clicked.connect(self._on_scale_changed)
        self.btn_lum.clicked.connect(self._on_chan_changed)
        self.btn_rgb.clicked.connect(self._on_chan_changed)

        self._update_styles(True)

    def attach_histogram(self, hist_widget):
        """Связывает шапку с полотном гистограммы."""
        self._hist_widget = hist_widget
        hist_widget.header = self
        self._sync_widget_state()

    def set_canvas_enabled(self, enabled: bool):
        """Блокирует переключатели при отсутствии изображения; кнопка [?] остается активной."""
        self.btn_full.setEnabled(enabled)
        self.btn_auto.setEnabled(enabled)
        self.btn_log.setEnabled(enabled)
        self.btn_lin.setEnabled(enabled)
        self.btn_lum.setEnabled(enabled)
        self.btn_rgb.setEnabled(enabled)
        self.btn_help.setEnabled(True)  # Кнопка справки всегда активна
        self._update_styles(enabled)

    def apply_theme(self, theme: dict):
        """Обновляет тему оформления согласно активному стейту A / B."""
        state = theme.get('name', 'A')
        self._active_state = state
        self._accent = '#FF9E0B' if state == 'B' else '#00E5FF'
        self._bright = theme.get('bright', '#00fff2' if state == 'A' else '#fbbf24')
        self._update_styles(self.btn_log.isEnabled())
        if self._hist_widget is not None:
            self._hist_widget.set_state_ab(state)

    def _update_styles(self, enabled: bool):
        accent = self._accent if enabled else '#334155'
        bright = self._bright if enabled else '#475569'

        btn_style_tpl = """
            QPushButton {{
                background: {bg};
                border: 1px solid {border};
                color: {color};
                font-size: 8.5px;
                font-weight: bold;
                border-radius: 3px;
                padding: 0;
            }}
            QPushButton:hover {{
                background: {hover_bg};
                color: {hover_color};
            }}
            QPushButton:disabled {{
                background: transparent;
                border: 1px solid transparent;
                color: #334155;
            }}
        """

        # Стиль для активных/неактивных сегментов
        for btn in [self.btn_full, self.btn_auto, self.btn_log, self.btn_lin, self.btn_lum, self.btn_rgb]:
            if btn.isChecked():
                btn.setStyleSheet(btn_style_tpl.format(
                    bg='#18202c' if enabled else '#131722',
                    border=accent,
                    color=accent,
                    hover_bg='#222d3d' if enabled else '#131722',
                    hover_color=bright
                ))
            else:
                btn.setStyleSheet(btn_style_tpl.format(
                    bg='transparent',
                    border='transparent',
                    color='#64748B' if enabled else '#334155',
                    hover_bg='#1a2230' if enabled else 'transparent',
                    hover_color='#94a3b8' if enabled else '#334155'
                ))

        # Стиль для круглой кнопки [?]
        self.btn_help.setStyleSheet(f"""
            QPushButton {{
                background-color: #18202c;
                border: 1px solid #334155;
                color: #94a3b8;
                font-size: 9.5px;
                font-weight: bold;
                border-radius: 9px;
                padding: 0;
            }}
            QPushButton:hover {{
                background-color: #242f42;
                border-color: {self._accent};
                color: {self._bright};
            }}
            QPushButton:pressed {{
                background-color: #2c3a52;
            }}
        """)

    def _on_roi_changed(self):
        self._update_styles(self.btn_full.isEnabled())
        if self._hist_widget is not None:
            mode = "AUTO" if self.btn_auto.isChecked() else "FULL"
            self._hist_widget.set_roi_mode(mode)

    def _on_scale_changed(self):
        self._update_styles(self.btn_log.isEnabled())
        if self._hist_widget is not None:
            mode = "LOG" if self.btn_log.isChecked() else "LIN"
            self._hist_widget.set_scale_mode(mode)

    def _on_chan_changed(self):
        self._update_styles(self.btn_lum.isEnabled())
        if self._hist_widget is not None:
            mode = "RGB" if self.btn_rgb.isChecked() else "LUM"
            self._hist_widget.set_channel_mode(mode)
            self.lbl_title.setText("RGB HISTOGRAM" if mode == "RGB" else "LUMINANCE HISTOGRAM")

    def _sync_widget_state(self):
        if self._hist_widget is not None:
            roi_mode = "AUTO" if self.btn_auto.isChecked() else "FULL"
            scale_mode = "LOG" if self.btn_log.isChecked() else "LIN"
            chan_mode = "RGB" if self.btn_rgb.isChecked() else "LUM"
            self._hist_widget.set_roi_mode(roi_mode)
            self._hist_widget.set_scale_mode(scale_mode)
            self._hist_widget.set_channel_mode(chan_mode)
            self._hist_widget.set_state_ab(self._active_state)

    def _open_help_dialog(self):
        try:
            from .dialogs import HistogramHelpDialog
            dlg = HistogramHelpDialog(self.window())
            dlg.exec()
        except Exception as e:
            QMessageBox.information(
                self,
                "Справка по гистограмме",
                f"Справка анализатора гистограммы:\n\n{e}"
            )


class AstroHistogramWidget(QWidget):
    """Профессиональный спектро-яркостный анализатор с LOG/LIN, RGB, LED-клиппингом и визирами."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(84)
        self.setMinimumWidth(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

        self.header = None
        self._roi_mode = "FULL"  # "FULL" или "AUTO"
        self._last_img = None
        self._scale_mode = "LOG"
        self._channel_mode = "LUM"
        self._active_state = "A"
        self._theme_accent = "#00E5FF"
        self._is_master_bypassed = False

        self._has_data = False
        self._is_rgb_image = False

        self._raw_hist_lum = np.zeros(256, dtype=np.float32)
        self._raw_hist_r = np.zeros(256, dtype=np.float32)
        self._raw_hist_g = np.zeros(256, dtype=np.float32)
        self._raw_hist_b = np.zeros(256, dtype=np.float32)

        self._hist_lum = np.zeros(256, dtype=np.float32)
        self._hist_r = np.zeros(256, dtype=np.float32)
        self._hist_g = np.zeros(256, dtype=np.float32)
        self._hist_b = np.zeros(256, dtype=np.float32)

        self._shadow_clip = False
        self._shadow_clip_pct = 0.0
        self._highlight_clip = False
        self._highlight_clip_pct = 0.0

        self._black_point_norm = 0.0
        self._black_point_enabled = False
        self._white_point_norm = 1.0
        self._white_point_enabled = False

    def clear(self):
        """Очищает данные при сбросе кадра."""
        self._has_data = False
        self._last_img = None
        self._shadow_clip = False
        self._shadow_clip_pct = 0.0
        self._highlight_clip = False
        self._highlight_clip_pct = 0.0
        self._raw_hist_lum.fill(0.0)
        self._raw_hist_r.fill(0.0)
        self._raw_hist_g.fill(0.0)
        self._raw_hist_b.fill(0.0)
        self._hist_lum.fill(0.0)
        self._hist_r.fill(0.0)
        self._hist_g.fill(0.0)
        self._hist_b.fill(0.0)
        self.update()

    def set_roi_mode(self, mode: str):
        """Устанавливает режим выборки данных ('FULL' или 'AUTO')."""
        if mode in ("FULL", "AUTO") and self._roi_mode != mode:
            self._roi_mode = mode
            if self._last_img is not None:
                self.update_histogram(self._last_img)

    def get_roi_mode(self) -> str:
        """Возвращает текущий режим выборки ('FULL' или 'AUTO')."""
        return self._roi_mode

    def set_frame(self, img: np.ndarray):
        """Псевдоним для update_histogram."""
        self.update_histogram(img)

    def update_histogram(self, img: np.ndarray):
        """Вычисляет 256-биновые гистограммы и проверяет клиппинг в тенях и светах."""
        if img is None or img.size == 0:
            return

        self._last_img = img

        total_px = img.shape[0] * img.shape[1]
        if total_px <= 0:
            return

        # Быстрый даунсэмплинг при необходимости (>65536 px)
        if total_px > 65536:
            step = max(1, int(np.sqrt(total_px / 65536)))
            sample = img[::step, ::step]
        else:
            sample = img

        is_float = np.issubdtype(sample.dtype, np.floating)

        if sample.ndim == 3 and sample.shape[2] >= 3:
            self._is_rgb_image = True
            r = sample[:, :, 0]
            g = sample[:, :, 1]
            b = sample[:, :, 2]

            if is_float:
                r_255 = np.clip(r * 255.0, 0.0, 255.0)
                g_255 = np.clip(g * 255.0, 0.0, 255.0)
                b_255 = np.clip(b * 255.0, 0.0, 255.0)
            else:
                r_255 = r.astype(np.float32)
                g_255 = g.astype(np.float32)
                b_255 = b.astype(np.float32)

            # Rec.709 / ITU-R: Y = 0.2126*R + 0.7152*G + 0.0722*B
            lum_255 = 0.2126 * r_255 + 0.7152 * g_255 + 0.0722 * b_255
            lum_255 = np.clip(lum_255, 0.0, 255.0)
        else:
            self._is_rgb_image = False
            mono = sample[:, :, 0] if sample.ndim == 3 else sample
            if is_float:
                lum_255 = np.clip(mono * 255.0, 0.0, 255.0)
            else:
                lum_255 = mono.astype(np.float32)
            r_255 = lum_255
            g_255 = lum_255
            b_255 = lum_255

        # Выделение полезного сигнала объекта в режиме AUTO (Auto-Mask ROI)
        use_mask = False
        mask = None
        if self._roi_mode == "AUTO":
            gray_u8 = np.clip(lum_255, 0.0, 255.0).astype(np.uint8)
            try:
                # Отсечение темнового фона космоса (Otsu + проверка шумового порога матрицы)
                thresh_val, thresh = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if thresh_val < 15:
                    _, thresh = cv2.threshold(gray_u8, 15, 255, cv2.THRESH_BINARY)

                # Нахождение связной области / контура диска планеты
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    if area >= 20:
                        mask = np.zeros(gray_u8.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [c], -1, 255, thickness=-1)
                        if np.any(mask > 0):
                            use_mask = True
            except Exception:
                use_mask = False

        if use_mask and mask is not None:
            roi = mask > 0
            lum_data = lum_255[roi]
            if lum_data.size == 0:
                lum_data = lum_255.ravel()
                if self._is_rgb_image:
                    r_data = r_255.ravel()
                    g_data = g_255.ravel()
                    b_data = b_255.ravel()
            else:
                if self._is_rgb_image:
                    r_data = r_255[roi]
                    g_data = g_255[roi]
                    b_data = b_255[roi]
        else:
            lum_data = lum_255.ravel()
            if self._is_rgb_image:
                r_data = r_255.ravel()
                g_data = g_255.ravel()
                b_data = b_255.ravel()

        if self._is_rgb_image:
            self._raw_hist_r, _ = np.histogram(r_data, bins=256, range=(0, 256))
            self._raw_hist_g, _ = np.histogram(g_data, bins=256, range=(0, 256))
            self._raw_hist_b, _ = np.histogram(b_data, bins=256, range=(0, 256))
            self._raw_hist_lum, _ = np.histogram(lum_data, bins=256, range=(0, 256))

            n_samples = max(1, lum_data.size)
            # Shadow clip: строго 0 (порог 0.05% кадра)
            shadow_px = int(np.sum(lum_data <= 0.05))
            self._shadow_clip_pct = (shadow_px / float(n_samples)) * 100.0
            self._shadow_clip = self._shadow_clip_pct > 0.05

            # Highlight clip: пережог 255
            hl_px = int(np.sum(lum_data >= 254.95))
            hl_px_rgb = max(int(np.sum(r_data >= 254.95)), int(np.sum(g_data >= 254.95)), int(np.sum(b_data >= 254.95)))
            effective_hl_px = max(hl_px, hl_px_rgb)
            self._highlight_clip_pct = (effective_hl_px / float(n_samples)) * 100.0
            self._highlight_clip = effective_hl_px > 0
        else:
            self._raw_hist_lum, _ = np.histogram(lum_data, bins=256, range=(0, 256))
            self._raw_hist_r = self._raw_hist_lum.copy()
            self._raw_hist_g = self._raw_hist_lum.copy()
            self._raw_hist_b = self._raw_hist_lum.copy()

            n_samples = max(1, lum_data.size)
            shadow_px = int(np.sum(lum_data <= 0.05))
            self._shadow_clip_pct = (shadow_px / float(n_samples)) * 100.0
            self._shadow_clip = self._shadow_clip_pct > 0.05

            hl_px = int(np.sum(lum_data >= 254.95))
            self._highlight_clip_pct = (hl_px / float(n_samples)) * 100.0
            self._highlight_clip = hl_px > 0

        self._has_data = True
        self._recompute_normalized_curves()
        self.update()

    def _recompute_normalized_curves(self):
        """Пересчитывает нормализацию Y-оси при смене LOG <-> LIN."""
        if not self._has_data:
            return

        max_lum = float(np.max(self._raw_hist_lum))
        if self._scale_mode == "LOG":
            denom_lum = float(np.log1p(max_lum + 1e-6))
            self._hist_lum = np.log1p(self._raw_hist_lum) / denom_lum if denom_lum > 0 else np.zeros(256, dtype=np.float32)
        else:
            self._hist_lum = self._raw_hist_lum / (max_lum + 1e-6) if max_lum > 0 else np.zeros(256, dtype=np.float32)

        max_rgb = float(max(np.max(self._raw_hist_r), np.max(self._raw_hist_g), np.max(self._raw_hist_b), 1.0))
        if self._scale_mode == "LOG":
            denom_rgb = float(np.log1p(max_rgb + 1e-6))
            self._hist_r = np.log1p(self._raw_hist_r) / denom_rgb if denom_rgb > 0 else np.zeros(256, dtype=np.float32)
            self._hist_g = np.log1p(self._raw_hist_g) / denom_rgb if denom_rgb > 0 else np.zeros(256, dtype=np.float32)
            self._hist_b = np.log1p(self._raw_hist_b) / denom_rgb if denom_rgb > 0 else np.zeros(256, dtype=np.float32)
        else:
            self._hist_r = self._raw_hist_r / (max_rgb + 1e-6)
            self._hist_g = self._raw_hist_g / (max_rgb + 1e-6)
            self._hist_b = self._raw_hist_b / (max_rgb + 1e-6)

    def set_scale_mode(self, mode: str):
        """Устанавливает масштаб Y-оси ('LOG' или 'LIN')."""
        if mode in ("LOG", "LIN") and self._scale_mode != mode:
            self._scale_mode = mode
            self._recompute_normalized_curves()
            self.update()

    def set_channel_mode(self, mode: str):
        """Устанавливает режим отображения каналов ('LUM' или 'RGB')."""
        if mode in ("LUM", "RGB") and self._channel_mode != mode:
            self._channel_mode = mode
            self.update()

    def set_state_ab(self, state: str):
        """Синхронизирует акцентный цвет со State A (#00E5FF) / State B (#FF9E0B)."""
        self._active_state = state
        self._theme_accent = "#FF9E0B" if state == "B" else "#00E5FF"
        self.update()

    def set_master_bypass(self, active: bool):
        """Переключает цвет кривой в приглушенный серо-голубой (#64748B) при Master Bypass."""
        self._is_master_bypassed = bool(active)
        self.update()

    def set_cut_points(self, black_val: float, gamma_val: float, black_on: bool = True, gamma_on: bool = True):
        """Синхронизирует позиции визиров Black Point и White Point с Insert 3."""
        self._black_point_norm = float(np.clip(black_val, 0.0, 1.0))
        self._black_point_enabled = bool(black_on)

        # При gamma >= 1.0 света сжимаются влево: white_pos = 1.0 / gamma
        if gamma_val >= 1.0:
            self._white_point_norm = float(np.clip(1.0 / max(gamma_val, 0.01), 0.1, 1.0))
        else:
            self._white_point_norm = 1.0
        self._white_point_enabled = bool(gamma_on)
        self.update()

    def apply_theme(self, theme: dict):
        state = theme.get('name', 'A')
        self.set_state_ab(state)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = float(self.width()), float(self.height())
        y_top = 4.0
        y_base = h - 16.0
        h_plot = max(y_base - y_top, 10.0)

        # 1. Фоновая подложка и рамка
        clip_path = QPainterPath()
        clip_path.addRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 6.0, 6.0)
        painter.fillPath(clip_path, QBrush(QColor("#10131A")))
        painter.strokePath(clip_path, QPen(QColor("#1E293B"), 1.0))
        painter.setClipPath(clip_path)

        # 2. Разметка сетки графика: 64, 128 (Midtones), 192, 255
        grid_pen = QPen(QColor("#1E2538"), 1.0, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        for bin_idx in (64, 128, 192):
            gx = (bin_idx / 255.0) * w
            painter.drawLine(int(gx), int(y_top), int(gx), int(y_base))

        # Базовая линия нулевого уровня
        painter.setPen(QPen(QColor("#1E2538"), 1.0, Qt.PenStyle.SolidLine))
        painter.drawLine(0, int(y_base), int(w), int(y_base))

        # 3. Отрисовка кривых гистограммы (если кадр загружен и виджет активен)
        if self._has_data and self.isEnabled():
            if self._channel_mode == "LUM":
                # Режим LUM: Rec.709 интегральная светимость
                if self._is_master_bypassed:
                    accent_color = QColor("#64748B")
                else:
                    accent_color = QColor(self._theme_accent)

                path = QPainterPath()
                path.moveTo(0.0, y_base)
                for i in range(256):
                    x = (i / 255.0) * w
                    y = y_base - float(self._hist_lum[i]) * h_plot
                    path.lineTo(x, y)
                path.lineTo(w, y_base)
                path.closeSubpath()

                # Неоновый DAW-градиент заливки (~38% сверху, ~5% у основания)
                grad = QLinearGradient(0.0, y_top, 0.0, y_base)
                grad.setColorAt(0.0, QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 96))
                grad.setColorAt(1.0, QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 12))
                painter.fillPath(path, QBrush(grad))

                # Контурная линия графика 1.5px
                outline = QPainterPath()
                outline.moveTo(0.0, y_base - float(self._hist_lum[0]) * h_plot)
                for i in range(1, 256):
                    outline.lineTo((i / 255.0) * w, y_base - float(self._hist_lum[i]) * h_plot)
                painter.strokePath(outline, QPen(accent_color, 1.5))

            else:
                # Режим RGB: независимые Red (#EF4444), Green (#10B981), Blue (#3B82F6)
                channels = [
                    (QColor("#EF4444"), self._hist_r),
                    (QColor("#10B981"), self._hist_g),
                    (QColor("#3B82F6"), self._hist_b)
                ]

                # Режим наложения Screen для аддитивного смешивания цветов
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Screen)

                for col, hist_arr in channels:
                    path = QPainterPath()
                    path.moveTo(0.0, y_base)
                    for i in range(256):
                        x = (i / 255.0) * w
                        y = y_base - float(hist_arr[i]) * h_plot
                        path.lineTo(x, y)
                    path.lineTo(w, y_base)
                    path.closeSubpath()

                    # Полупрозрачная заливка 25%
                    painter.fillPath(path, QBrush(QColor(col.red(), col.green(), col.blue(), 64)))

                    outline = QPainterPath()
                    outline.moveTo(0.0, y_base - float(hist_arr[0]) * h_plot)
                    for i in range(1, 256):
                        outline.lineTo((i / 255.0) * w, y_base - float(hist_arr[i]) * h_plot)
                    painter.strokePath(outline, QPen(col, 1.5))

                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # 4. Визиры Black Point и White Point
        if self.isEnabled():
            if self._black_point_enabled and self._black_point_norm > 0.001:
                bx = float(self._black_point_norm) * w
                painter.setPen(QPen(QColor("#64748B"), 1.0, Qt.PenStyle.DashLine))
                painter.drawLine(int(bx), int(y_top), int(bx), int(y_base))

            if self._white_point_enabled and self._white_point_norm < 0.999:
                wx = float(self._white_point_norm) * w
                painter.setPen(QPen(QColor("#94A3B8"), 1.0, Qt.PenStyle.DashLine))
                painter.drawLine(int(wx), int(y_top), int(wx), int(y_base))

        # 5. Светодиодные индикаторы клиппинга
        # Левый LED: Under-exposure / Shadows Clip (x=12, y=12, dot 6x6)
        lx, ly = 12.0, 12.0
        if self.isEnabled() and self._has_data and self._shadow_clip:
            halo = QRadialGradient(lx, ly, 8.0)
            halo.setColorAt(0.0, QColor(0, 229, 255, 180))
            halo.setColorAt(0.5, QColor(56, 189, 248, 80))
            halo.setColorAt(1.0, QColor(0, 229, 255, 0))
            painter.fillRect(QRectF(lx - 8, ly - 8, 16, 16), QBrush(halo))

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#00E5FF")))
            painter.drawEllipse(QRectF(lx - 3, ly - 3, 6, 6))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#334155")))
            painter.drawEllipse(QRectF(lx - 3, ly - 3, 6, 6))

        # Правый LED: Over-exposure / Highlights Clip (x=w-12, y=12, dot 6x6)
        rx, ry = w - 12.0, 12.0
        if self.isEnabled() and self._has_data and self._highlight_clip:
            halo = QRadialGradient(rx, ry, 8.0)
            halo.setColorAt(0.0, QColor(239, 68, 68, 180))
            halo.setColorAt(0.5, QColor(239, 68, 68, 80))
            halo.setColorAt(1.0, QColor(239, 68, 68, 0))
            painter.fillRect(QRectF(rx - 8, ry - 8, 16, 16), QBrush(halo))

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#EF4444")))
            painter.drawEllipse(QRectF(rx - 3, ry - 3, 6, 6))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor("#334155")))
            painter.drawEllipse(QRectF(rx - 3, ry - 3, 6, 6))

        # 6. Подписи шкалы яркости
        painter.setPen(QPen(QColor("#475569")))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(6, int(h - 3), "0 (Shadows)")
        painter.drawText(int(w * 0.50) - 34, int(h - 3), "128 (Midtones)")
        painter.drawText(int(w - 70), int(h - 3), "255 (Peaks)")

    def event(self, event):
        """Интерактивные тултипы для светодиодов клиппинга и рабочей области."""
        if event.type() == QEvent.Type.ToolTip:
            pos = event.pos()
            x, y = pos.x(), pos.y()
            w = self.width()

            if x <= 28 and y <= 24:
                if self._shadow_clip:
                    tip = f"Shadow Clipping: {self._shadow_clip_pct:.2f}% пикселей потеряно в чистом нуле (провал деталей фона)"
                else:
                    tip = "Shadows: No clipping (чистый фон сохранён)"
                QToolTip.showText(event.globalPos(), tip, self)
                return True
            elif x >= w - 28 and y <= 24:
                if self._highlight_clip:
                    tip = f"Highlight Clipping (Пережог): {self._highlight_clip_pct:.2f}% пикселей насыщены до 255 (потеря текстуры в ярких зонах / кратерах)"
                else:
                    tip = "Highlights: No clipping (динамический диапазон в норме)"
                QToolTip.showText(event.globalPos(), tip, self)
                return True
            else:
                roi_str = "Auto-Mask ROI" if self._roi_mode == "AUTO" else "Full Frame"
                scale_str = "LOG" if self._scale_mode == "LOG" else "Linear"
                chan_str = "RGB" if self._channel_mode == "RGB" else "Rec.709 Lum"
                tip = f"Спектро-яркостный анализатор ({roi_str}, {scale_str}, {chan_str})"
                QToolTip.showText(event.globalPos(), tip, self)
                return True

        return super().event(event)


# =============================================================================
# РУЧКА-ПЛАНКА СВОРАЧИВАНИЯ И РЕСАЙЗА ПРАВОГО РЭКА (RACK TAB HANDLE)
# =============================================================================

class RackTabHandle(QFrame):
    """
    Левая вертикальная планка-язычок правого рэка инсертов (ширина 14–16 px).
    Служит ручкой интерактивного изменения ширины (Splitter Handle) и
    содержит кнопку-шеврон для сворачивания/разворачивания рэка инсертов.
    """
    toggle_requested = pyqtSignal()
    width_dragged = pyqtSignal(int)  # delta в пикселях

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rackTabHandle")
        self.setFixedWidth(16)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._collapsed = False
        self._dragging = False
        self._drag_start_x = 0.0
        self._initial_panel_width = 350

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addStretch(1)

        self.collapse_btn = QPushButton("⯈", self)
        self.collapse_btn.setObjectName("rackCollapseBtn")
        self.collapse_btn.setFlat(True)
        self.collapse_btn.setFixedSize(16, 36)
        self.collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_btn.setToolTip("Свернуть панель инсертов")
        self.collapse_btn.clicked.connect(self._on_btn_clicked)
        self.btn_toggle = self.collapse_btn  # Обратная совместимость
        layout.addWidget(self.collapse_btn)
        layout.addStretch(1)

        self.update_collapse_visual_state(is_collapsed=False)

    def set_collapsed(self, collapsed: bool):
        self._collapsed = bool(collapsed)
        self.update_collapse_visual_state(is_collapsed=self._collapsed)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def update_collapse_visual_state(self, is_collapsed: bool = False):
        self._collapsed = bool(is_collapsed)
        if self._collapsed:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setStyleSheet("""
                QFrame#rackTabHandle {
                    background-color: #101218;
                    border-left: 2px solid #00e5ff;
                    border-right: none;
                    border-top: none;
                    border-bottom: none;
                }
            """)
            self.collapse_btn.setText("⯇")
            self.collapse_btn.setToolTip("Развернуть панель инсертов")
            self.collapse_btn.setStyleSheet("""
                QPushButton#rackCollapseBtn {
                    background: transparent;
                    color: #64748b;
                    border: none;
                    font-size: 10px;
                    padding: 0;
                    margin: 0;
                }
                QPushButton#rackCollapseBtn:hover {
                    color: #00e5ff;
                }
            """)
        else:
            self.setCursor(Qt.CursorShape.SplitHCursor)
            self.setStyleSheet("""
                QFrame#rackTabHandle {
                    background-color: #101218;
                    border-left: 1px solid #1a202c;
                    border-right: none;
                    border-top: none;
                    border-bottom: none;
                }
            """)
            self.collapse_btn.setText("⯈")
            self.collapse_btn.setToolTip("Свернуть панель инсертов")
            self.collapse_btn.setStyleSheet("""
                QPushButton#rackCollapseBtn {
                    background: transparent;
                    color: #64748b;
                    border: none;
                    font-size: 10px;
                    padding: 0;
                    margin: 0;
                }
                QPushButton#rackCollapseBtn:hover {
                    color: #00e5ff;
                }
            """)

    def _update_appearance(self):
        self.update_collapse_visual_state(is_collapsed=self._collapsed)

    def _on_btn_clicked(self):
        self.toggle_requested.emit()

    def enterEvent(self, event):
        super().enterEvent(event)
        if not self._collapsed:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._collapsed:
                self.toggle_requested.emit()
            else:
                self._dragging = True
                self._drag_start_x = event.globalPosition().x()
                panel = self.parentWidget()
                self._initial_panel_width = panel.width() if panel else 340
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._collapsed:
            if not self._dragging:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                delta = int(round(event.globalPosition().x() - self._drag_start_x))
                self.width_dragged.emit(delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
        super().mouseReleaseEvent(event)


