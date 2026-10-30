"""Диалоговые окна приложения AstroRack DAW."""
import os
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QWidget, QScrollArea
)
from PyQt6.QtGui import QFont, QDesktopServices, QPixmap, QIcon
from utils.file_io import get_resource_path


class AboutDialog(QDialog):
    """Кастомное модальное окно 'О программе' в строгом тёмном стиле DAW-платформы."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("О программе — AstroRack DAW")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(620, 680)

        icon_256_path = get_resource_path(os.path.join("assets", "icons", "icon_256x256.png"))
        icon_64_path = get_resource_path(os.path.join("assets", "icons", "icon_64x64.png"))
        ico_path = get_resource_path(os.path.join("assets", "app_icon.ico"))

        if os.path.exists(icon_256_path):
            self.setWindowIcon(QIcon(icon_256_path))
        elif os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))

        # Центрирование над родительским окном
        if parent:
            geo = parent.geometry()
            self.move(
                geo.x() + (geo.width() - 620) // 2,
                geo.y() + (geo.height() - 680) // 2
            )

        self.setStyleSheet("""
            AboutDialog {
                background-color: #161922;
                border: 1px solid #00adb5;
                border-radius: 8px;
            }
            QLabel {
                background: transparent;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        # 1. Шапка: Логотип 64x64, Версия и кнопка закрытия [✕]
        header_row = QHBoxLayout()
        header_row.setSpacing(14)

        self.lbl_icon = QLabel()
        self.lbl_icon.setFixedSize(64, 64)
        self.lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if os.path.isfile(icon_64_path):
            self.lbl_icon.setPixmap(QPixmap(icon_64_path))
        else:
            self.lbl_icon.setText("🔭")
            self.lbl_icon.setStyleSheet("font-size: 32px; color: #00fff2;")
        header_row.addWidget(self.lbl_icon)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        lbl_title = QLabel("AstroRack DAW v8.3.6")
        lbl_title.setStyleSheet("""
            font-size: 20px;
            font-weight: 800;
            color: #f1f5f9;
            letter-spacing: 0.5px;
        """)
        title_col.addWidget(lbl_title)

        lbl_sub = QLabel("Lunar, Planetary & Deep-Sky Image Mastering Platform")
        lbl_sub.setStyleSheet("font-size: 11px; color: #94a3b8; font-weight: 500;")
        title_col.addWidget(lbl_sub)
        header_row.addLayout(title_col)

        header_row.addStretch()

        btn_close_x = QPushButton("✕")
        btn_close_x.setFixedSize(26, 26)
        btn_close_x.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #94a3b8;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2e384d;
                color: #ffffff;
            }
        """)
        btn_close_x.clicked.connect(self.reject)
        header_row.addWidget(btn_close_x)

        layout.addLayout(header_row)

        # Разделитель
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #222a3d; color: #222a3d; max-height: 1px;")
        layout.addWidget(sep)

        # 2. Манифест свободы и лицензия
        manifesto_frame = QFrame()
        manifesto_frame.setStyleSheet("""
            QFrame {
                background-color: #0F1117;
                border: 1px solid #1E293B;
                border-radius: 8px;
            }
        """)
        mf_layout = QVBoxLayout(manifesto_frame)
        mf_layout.setContentsMargins(16, 14, 16, 14)
        mf_layout.setSpacing(8)

        lbl_mf_title = QLabel("✦ Манифест свободы & Лицензия")
        lbl_mf_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #00fff2;")
        mf_layout.addWidget(lbl_mf_title)

        lbl_mf_text = QLabel(
            "Лицензия: Public Domain / Zero Rights Reserved.\n\n"
            "Софт создан для сообщества любителей астрономии и астрофотографии. "
            "Свободно для личного и коммерческого использования без ограничений."
        )
        lbl_mf_text.setWordWrap(True)
        lbl_mf_text.setStyleSheet("font-size: 11.5px; color: #cbd5e1; line-height: 145%;")
        mf_layout.addWidget(lbl_mf_text)
        layout.addWidget(manifesto_frame)

        # 3. Использованный технологический стек
        stack_frame = QFrame()
        stack_frame.setStyleSheet("""
            QFrame {
                background-color: #12151f;
                border: 1px solid #1E293B;
                border-radius: 8px;
            }
        """)
        st_layout = QVBoxLayout(stack_frame)
        st_layout.setContentsMargins(16, 14, 16, 14)
        st_layout.setSpacing(8)

        lbl_stack_title = QLabel("⚙ Технологический стек ядра")
        lbl_stack_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #94a3b8;")
        st_layout.addWidget(lbl_stack_title)

        badges_row = QHBoxLayout()
        badges_row.setSpacing(8)
        techs = [
            ("PyQt6", "#38bdf8"),
            ("OpenCV", "#4ade80"),
            ("NumPy", "#fbbf24"),
            ("SciPy", "#f43f5e"),
            ("Matplotlib", "#a78bfa")
        ]
        for name, col in techs:
            b = QLabel(name)
            b.setStyleSheet(f"""
                background-color: #1c2233;
                border: 1px solid #2d374d;
                border-radius: 4px;
                color: {col};
                font-size: 10.5px;
                font-weight: bold;
                padding: 4px 10px;
            """)
            badges_row.addWidget(b)
        badges_row.addStretch()
        st_layout.addLayout(badges_row)
        layout.addWidget(stack_frame)

        layout.addStretch()

        # 4. Инженерная плашка телеметрии наземной станции (Astana Observatory Badge)
        station_badge = QFrame()
        station_badge.setStyleSheet("""
            QFrame {
                background-color: #10131A;
                border: 1px solid #1E293B;
                border-radius: 6px;
            }
        """)
        sb_layout = QHBoxLayout(station_badge)
        sb_layout.setContentsMargins(12, 8, 12, 8)
        sb_layout.setSpacing(8)

        lbl_sat_icon = QLabel("📡")
        lbl_sat_icon.setStyleSheet("color: #00E5FF; font-size: 13px; background: transparent; border: none;")
        sb_layout.addWidget(lbl_sat_icon)

        lbl_station_text = QLabel("Ground Station: Astana, KZ (51°08' N, 71°26' E) • Epoch J2026.10")
        lbl_station_text.setStyleSheet("""
            font-family: 'Consolas', 'JetBrains Mono', 'Courier New', monospace;
            font-size: 9pt;
            color: #64748B;
            background: transparent;
            border: none;
        """)
        sb_layout.addWidget(lbl_station_text)
        sb_layout.addStretch()
        layout.addWidget(station_badge)

        # 5. Нижние кнопки: GitHub и Закрыть
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)

        btn_github = QPushButton("★  GitHub: shablakovmihail-dev/AstroRack-DAW")
        btn_github.setFixedHeight(34)
        btn_github.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_github.setToolTip("https://github.com/shablakovmihail-dev/AstroRack-DAW")
        btn_github.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 229, 255, 0.08);
                border: 1px solid #00E5FF;
                border-radius: 6px;
                color: #00E5FF;
                font-size: 11px;
                font-weight: bold;
                padding: 0 14px;
            }
            QPushButton:hover {
                background-color: rgba(0, 229, 255, 0.20);
                border: 1px solid #00E5FF;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #00E5FF;
                border-color: #00E5FF;
                color: #0f1117;
            }
        """)
        btn_github.clicked.connect(self._open_github)
        bottom_row.addWidget(btn_github)

        bottom_row.addStretch()

        btn_ok = QPushButton("Закрыть [Esc]")
        btn_ok.setFixedHeight(34)
        btn_ok.setFixedWidth(110)
        btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #1e2433;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #cbd5e1;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #273044;
                border-color: #00adb5;
                color: #ffffff;
            }
        """)
        btn_ok.clicked.connect(self.accept)
        bottom_row.addWidget(btn_ok)

        layout.addLayout(bottom_row)

    def _open_github(self):
        QDesktopServices.openUrl(QUrl("https://github.com/shablakovmihail-dev/AstroRack-DAW"))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            event.accept()
            return
        super().keyPressEvent(event)


class ResetConfirmDialog(QDialog):
    """Кастомный модальный диалог подтверждения сброса в тёмной эстетике DAW."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Подтверждение сброса")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(420, 190)

        icon_path = get_resource_path(os.path.join("assets", "icons", "icon_32x32.png"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        if parent:
            geo = parent.geometry()
            self.move(
                geo.x() + (geo.width() - 420) // 2,
                geo.y() + (geo.height() - 190) // 2
            )

        self.setStyleSheet("""
            ResetConfirmDialog {
                background-color: #161922;
                border: 1px solid #1E293B;
                border-radius: 8px;
            }
            QLabel {
                background: transparent;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # 1. Шапка: Заголовок и кнопка закрытия
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        lbl_title = QLabel("Подтверждение сброса")
        lbl_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 700;
            color: #f1f5f9;
        """)
        header_row.addWidget(lbl_title)
        header_row.addStretch()

        btn_close = QPushButton("✕")
        btn_close.setFixedSize(22, 22)
        btn_close.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #64748B;
                font-size: 13px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #262B3D;
                color: #ffffff;
            }
        """)
        btn_close.clicked.connect(self.reject)
        header_row.addWidget(btn_close)
        layout.addLayout(header_row)

        # Разделитель
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #1E293B; max-height: 1px;")
        layout.addWidget(sep)

        # 2. Текст предупреждения
        lbl_text = QLabel(
            "Вы уверены, что хотите сбросить текущее изображение и параметры обработки? "
            "Все несохранённые изменения будут потеряны."
        )
        lbl_text.setWordWrap(True)
        lbl_text.setStyleSheet("""
            font-size: 12px;
            color: #cbd5e1;
            line-height: 140%;
        """)
        layout.addWidget(lbl_text)

        layout.addStretch()

        # 3. Кнопки действий: [ Сбросить ] и [ Отмена ]
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(10)
        buttons_row.addStretch()

        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setFixedHeight(32)
        self.btn_cancel.setFixedWidth(100)
        self.btn_cancel.setDefault(True)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #1e2433;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #cbd5e1;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #273044;
                border-color: #475569;
                color: #ffffff;
            }
            QPushButton:focus {
                border-color: #00adb5;
            }
        """)
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_reset = QPushButton("Сбросить")
        self.btn_reset.setFixedHeight(32)
        self.btn_reset.setFixedWidth(100)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #221418;
                border: 1px solid #7f1d1d;
                border-radius: 6px;
                color: #f87171;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #dc2626;
                border-color: #ef4444;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #991b1b;
                border-color: #b91c1c;
                color: #ffffff;
            }
        """)
        self.btn_reset.clicked.connect(self.accept)

        buttons_row.addWidget(self.btn_reset)
        buttons_row.addWidget(self.btn_cancel)
        layout.addLayout(buttons_row)

        self.btn_cancel.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            event.accept()
            return
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.btn_reset.hasFocus():
                self.accept()
            else:
                self.reject()
            event.accept()
            return
        super().keyPressEvent(event)


class HistogramHelpDialog(QDialog):
    """Кастомное модальное окно 'Справка по анализу гистограммы' в строгом тёмном стиле DAW."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка: Анализ гистограммы — AstroRack DAW")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedSize(580, 580)
        self._drag_pos = None

        icon_256_path = get_resource_path(os.path.join("assets", "icons", "icon_256x256.png"))
        icon_64_path = get_resource_path(os.path.join("assets", "icons", "icon_64x64.png"))
        ico_path = get_resource_path(os.path.join("assets", "app_icon.ico"))

        if os.path.exists(icon_256_path):
            self.setWindowIcon(QIcon(icon_256_path))
        elif os.path.exists(ico_path):
            self.setWindowIcon(QIcon(ico_path))

        if parent:
            geo = parent.geometry()
            self.move(
                geo.x() + (geo.width() - 580) // 2,
                geo.y() + (geo.height() - 580) // 2
            )

        self.setStyleSheet("""
            HistogramHelpDialog {
                background-color: #161922;
                border: 1px solid #1E293B;
                border-radius: 8px;
            }
            QLabel {
                background: transparent;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #11141d;
                width: 6px;
                border-radius: 3px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #2b3548;
                min-height: 20px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00adb5;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # 1. Шапка: Иконка, Заголовок и кнопка закрытия [✕]
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        lbl_icon = QLabel()
        lbl_icon.setFixedSize(36, 36)
        if os.path.exists(icon_64_path):
            pix = QPixmap(icon_64_path).scaled(36, 36, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl_icon.setPixmap(pix)
        else:
            lbl_icon.setText("📊")
            lbl_icon.setStyleSheet("font-size: 24px; color: #00fff2;")
        header_row.addWidget(lbl_icon)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(1)
        lbl_title = QLabel("Справка по анализу гистограммы")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #00fff2; letter-spacing: 0.5px;")
        lbl_sub = QLabel("ПРОФЕССИОНАЛЬНЫЙ АСТРОНОМИЧЕСКИЙ СПЕКТРОАНАЛИЗАТОР")
        lbl_sub.setStyleSheet("font-size: 9px; color: #94a3b8; font-weight: 600; letter-spacing: 1px;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_sub)
        header_row.addLayout(title_vbox, stretch=1)

        btn_close_x = QPushButton("✕")
        btn_close_x.setFixedSize(22, 22)
        btn_close_x.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close_x.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #64748B;
                font-size: 13px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #262B3D;
                color: #ffffff;
            }
        """)
        btn_close_x.clicked.connect(self.accept)
        header_row.addWidget(btn_close_x)
        layout.addLayout(header_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #1E293B; max-height: 1px;")
        layout.addWidget(sep)

        # 2. Область карточек со скроллом
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        cards_vbox = QVBoxLayout(scroll_content)
        cards_vbox.setContentsMargins(0, 4, 8, 4)
        cards_vbox.setSpacing(10)

        # Блок 1: Логарифмическая шкала (LOG vs LIN)
        card1 = self._create_card(
            title="1. Логарифмическая шкала (LOG vs LIN)",
            badge="LOG / LIN",
            badge_color="#00E5FF",
            text=(
                "<b>Почему для космоса важен LOG:</b> В планетной и лунной астрофотографии до 80–95% площади кадра "
                "занимает черный космос, что создает экстремальный спайк (колоссальный пик) в районе нулевых значений "
                "яркости. На обычной линейной шкале (<b>LIN</b>) весь полезный динамический диапазон диска планеты "
                "или лунных морей сплющивается в нечитаемую линию толщиной в 1 пиксель.<br><br>"
                "<b>LOG-режим</b> логарифмирует вертикальную ось Y: <i>y = ln(1 + hist) / ln(1 + max)</i>. "
                "Это эффективно вытягивает полутона атмосферных поясов Юпитера, колец Сатурна и морей Луны "
                "для точного визуального контроля динамического диапазона без ослепления фоновым спайком."
            )
        )
        cards_vbox.addWidget(card1)

        # Блок 2: Чтение каналов (LUM vs RGB)
        card2 = self._create_card(
            title="2. Чтение каналов (LUM vs RGB)",
            badge="LUM | RGB",
            badge_color="#38BDF8",
            text=(
                "<b>Режим LUM (Luminance):</b> Рассчитывает интегральную спектральную яркость по вещательному "
                "стандарту Rec.709 (<i>Y = 0.2126·R + 0.7152·G + 0.0722·B</i>). График заливается акцентным цветом "
                "текущего активного стейта: неоновый циан (#00E5FF) для State [ A ] и янтарный (#FF9E0B) для State [ B ].<br><br>"
                "<b>Режим RGB:</b> Отрисовывает три независимых кривых контура: <font color='#EF4444'><b>Red</b></font>, "
                "<font color='#10B981'><b>Green</b></font> и <font color='#3B82F6'><b>Blue</b></font> с наложением. "
                "Несовпадение пиков R, G и B на нейтрально-серых участках (например, лунном грунте) указывает на сбитый "
                "баланс белого, а расхождение синего влево и красного вправо на границах диска сигнализирует об атмосферной "
                "дисперсии (требуется Auto RGB Align в Insert 4)."
            )
        )
        cards_vbox.addWidget(card2)

        # Блок 3: Индикаторы клиппинга и радарный поиск зон
        card3 = self._create_card(
            title="3. Индикаторы клиппинга и радарная локализация",
            badge="SONAR CLIPPING",
            badge_color="#ff1744",
            text=(
                "<b>Индикатор теней слева (Shadows / Underflow):</b> Светодиод и кнопка счётчика "
                "с акцентным электрическим лазурным цветом (<font color='#00b0ff'><b>#00b0ff</b></font>). "
                "Показывает точный процент и физическое количество пикселей с яркостью ≤ 0.001 (чистый ноль). "
                "Предупреждает о провале фона космоса и безвозвратной потере слабых деталей (пепельный свет Луны, внешние кольца Сатурна).<br><br>"
                "<b>Индикатор светов справа (Highlights / Overflow / Peaks):</b> Светодиод и кнопка счётчика "
                "с акцентным неоновым красным цветом (<font color='#ff1744'><b>#ff1744</b></font>). "
                "Показывает зоны насыщения матрицы (яркость ≥ 0.999 / 255). В этих участках информация о рельефе полностью утрачена "
                "(выгорание центральных пиков кратеров или освещённого лимба).<br><br>"
                "<b>Интерактивный радарный поиск (Sonar Ping Overlay):</b> Нажмите на любой из индикаторов, "
                "чтобы включить интерактивную подсветку на холсте — проблемные участки подсвечиваются полупрозрачной маской "
                "и пульсирующими концентрическими радарными волнами вокруг ключевых центроидов. "
                "Повторный клик по индикатору или клавиша <b>Esc</b> отключают подсветку."
            )
        )
        cards_vbox.addWidget(card3)

        # Блок 4: Маркеры отсечения диапазона (Визиры)
        card4 = self._create_card(
            title="4. Маркеры отсечения диапазона (Визиры)",
            badge="VISORS",
            badge_color="#94A3B8",
            text=(
                "Вертикальные пунктирные линии на полотне гистограммы наглядно показывают границы динамического диапазона, "
                "заданные в <b>Insert 3 (Tone & Contrast)</b>, скользя в реальном времени при движении ползунков:<br>"
                "• <b>Левый визир (Black Point):</b> Пунктир (#64748B), привязан к слайдеру <i>Black Level (Sky Darkening)</i>.<br>"
                "• <b>Правый визир (White Point):</b> Пунктир (#94A3B8), привязан к слайдеру <i>White Point / Gamma</i>.<br><br>"
                "<b>Рекомендация:</b> Подводите маркер Black Point вплотную к левому склону горба сигнала объекта, "
                "не задевая полезную информацию, и удерживайте правый край гистограммы в пределах 220–240 единиц без касания "
                "правой стенки (255)."
            )
        )
        cards_vbox.addWidget(card4)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, stretch=1)

        # 3. Кнопка внизу: [ Понятно / Закрыть ]
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()

        self.btn_ok = QPushButton("Понятно / Закрыть")
        self.btn_ok.setFixedHeight(30)
        self.btn_ok.setMinimumWidth(150)
        self.btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ok.setDefault(True)
        self.btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #1e2433;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #cbd5e1;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 16px;
            }
            QPushButton:hover {
                background-color: #273044;
                border-color: #00adb5;
                color: #ffffff;
            }
            QPushButton:focus {
                border-color: #00adb5;
            }
        """)
        self.btn_ok.clicked.connect(self.accept)
        bottom_row.addWidget(self.btn_ok)
        layout.addLayout(bottom_row)

        self.btn_ok.setFocus()

    def _create_card(self, title: str, badge: str, badge_color: str, text: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #12151D;
                border: 1px solid #1E293B;
                border-radius: 6px;
            }
        """)
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(12, 10, 12, 10)
        vbox.setSpacing(6)

        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        lbl_t = QLabel(title)
        lbl_t.setStyleSheet(f"font-size: 11.5px; font-weight: 700; color: {badge_color};")
        hdr.addWidget(lbl_t)
        hdr.addStretch()

        lbl_b = QLabel(badge)
        lbl_b.setStyleSheet(f"""
            font-size: 9px;
            font-weight: 700;
            color: {badge_color};
            background-color: #18202c;
            border: 1px solid {badge_color};
            border-radius: 3px;
            padding: 1px 6px;
        """)
        hdr.addWidget(lbl_b)
        vbox.addLayout(hdr)

        lbl_body = QLabel(text)
        lbl_body.setWordWrap(True)
        lbl_body.setStyleSheet("font-size: 10.5px; color: #cbd5e1; line-height: 140%;")
        vbox.addWidget(lbl_body)

        return card

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
            event.accept()
            return
        super().keyPressEvent(event)

