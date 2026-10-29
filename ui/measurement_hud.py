"""
Компактная нижняя панель измерений (Measurement Bottom Bar / Shelf) для AstroRack DAW.
Изолированная панель управления с компактными иконками, горизонтальным скроллом безопасности,
верхней шторкой-язычком сворачивания/разворачивания и контекстным выводом параметров.
"""

from typing import Optional
import math
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QRadioButton, QButtonGroup,
    QVBoxLayout, QHBoxLayout, QDialog, QTextBrowser, QSizePolicy, QToolButton,
    QScrollArea
)
from PyQt6.QtGui import QFont, QColor


class MeasurementHelpDialog(QDialog):
    """Обучающий справочник по сферической коррекции и замеру кратеров Луны."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справочник: Сферическая коррекция и замер кратеров Луны")
        self.resize(600, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("🔭 Геометрическая оптика и сферическая проекция Луны")
        title.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setStyleSheet("""
            QTextBrowser {
                background-color: #12141a;
                color: #dcdce6;
                border: 1px solid #282c3c;
                border-radius: 6px;
                padding: 12px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                line-height: 1.5;
            }
        """)

        help_html = """
        <h3 style="color: #00e5ff; margin-top: 0;">1. Проблема ракурсного перспективного сжатия (cos &theta;)</h3>
        <p>Луна является шарообразным телом радиусом <b>R &approx; 1737.4 км</b>. Для земного наблюдателя видимый диск
        представляет собой ортографическую проекцию сферы на картинную плоскость матрицы камеры.</p>
        <p>Круглые ударные кратеры и элементы рельефа, удалённые от субземной точки (центра видимого диска Луны),
        видны под косым углом. В картинной плоскости они испытывают перспективное сжатие вдоль направления к лимбу
        с коэффициентом <code>cos &theta;</code>, превращаясь в эллипсы:</p>
        <ul>
            <li><b>&theta; = 0&deg;</b> (центр диска) &rarr; <code>cos &theta; = 1.0</code>: кратеры видны круглыми, ракурсных искажений нет.</li>
            <li><b>&theta; = 60&deg;</b> (2/3 радиуса к краю) &rarr; <code>cos &theta; = 0.5</code>: видимый радиальный размер сжат ровно в 2 раза! Обычная прямая линейка занизит размер вдвое.</li>
            <li><b>&theta; &rarr; 90&deg;</b> (лимб) &rarr; <code>cos &theta; &rarr; 0</code>: структуры превращаются в тонкие полоски на горизонте.</li>
        </ul>

        <h3 style="color: #00e5ff;">2. Инструмент «Moon Disk Alignment» (Привязка диска Луны)</h3>
        <p>Служит для автоматического определения глобальных сферических координат на всем снимке:</p>
        <ul>
            <li><b>Рамка (1:1 Bounding Box)</b>: растяните рамку по лимбу Луны с сохранением пропорций 1:1.
            Маркеры трансформации позволяют перемещать и масштабировать диск.</li>
            <li><b>3 Точки на лимбе</b>: укажите 3 точки на видимой дуге края Луны для аналитического расчёта центра и радиуса.</li>
            <li>Нажмите <b>[✓ Применить]</b> или <kbd>Enter</kbd> для фиксации калибровки.</li>
        </ul>
        <p>После калибровки линейка <b>Linear Ruler</b> автоматически рассчитывает
        как проекционный размер (<code>Proj</code>), так и <b>истинный диаметр на сфере (True = Proj / cos &theta;)</b>.</p>

        <h3 style="color: #00e5ff;">3. Инструмент «Crater Ellipse» (Автономный эллиптический замер)</h3>
        <p>Специально разработан для детальных снимков высокого разрешения, на которых лимб Луны целиком отсутствует:</p>
        <ul>
            <li><b>Шаг 1 (Главная ось 2a)</b>: протяните линию вдоль максимального поперечника кратера (параллельно ближайшему горизонту/лимбу).
            Поперечный размер параллелен картинной плоскости и <b>не искажен ракурсом</b>. Он равен <b>истинному диаметру (True Diameter)</b>!</li>
            <li><b>Шаг 2 (Малая ось 2b)</b>: переместите курсор перпендикулярно, подогнав эллипс под видимый вал кратера. Отношение <code>b / a</code>
            напрямую задает <code>cos &theta;</code> и локальный угол наклона лунной поверхности.</li>
        </ul>
        """
        browser.setHtml(help_html)
        layout.addWidget(browser)

        btn_close = QPushButton("Закрыть")
        btn_close.setFixedHeight(30)
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #00adb5;
                color: #111114;
                font-weight: bold;
                border-radius: 4px;
                padding: 4px 20px;
            }
            QPushButton:hover {
                background-color: #00fff2;
            }
        """)
        btn_close.clicked.connect(self.accept)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        self.setStyleSheet("""
            QDialog {
                background-color: #16161e;
                border: 1px solid #2e2e3e;
                border-radius: 8px;
            }
        """)


class MeasurementBottomBar(QFrame):
    """
    Компактная горизонтальная панель инструментов измерений (Measurement Bottom Bar / Shelf).
    Размещается под вьювером как фиксированная нижняя полка.
    Включает постоянную верхнюю шторку-язычок (14px) с кнопкой-шевроном для сворачивания/разворачивания
    и тело панели (46px) с горизонтальным скроллом безопасности.
    """

    mode_changed = pyqtSignal(str)              # "ruler", "disk", "ellipse"
    disk_method_changed = pyqtSignal(str)       # "center_radius", "three_points"
    apply_alignment_requested = pyqtSignal()
    reset_disk_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("measurementBottomBar")
        self.setFixedHeight(60)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._current_mode = "ruler"
        self._disk_method = "center_radius"
        self._is_collapsed = False
        self._is_calibrated = False
        self._calibrated_radius = 0.0
        self._pending_active = False
        self._last_proj_km = 0.0
        self._last_true_km = None
        self._last_tilt_deg = 0.0
        self._last_cos_theta = 1.0

        root_vbox = QVBoxLayout(self)
        root_vbox.setContentsMargins(0, 0, 0, 0)
        root_vbox.setSpacing(0)

        # ---------------- 0. ВЕРХНЯЯ ШТОРКА-ЯЗЫЧОК (Tab Bar / Top Handle) ----------------
        self.tab_handle = QFrame(self)
        self.tab_handle.setObjectName("shelfTabHandle")
        self.tab_handle.setFixedHeight(14)
        self.tab_handle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tab_handle.setStyleSheet("""
            QFrame#shelfTabHandle {
                background-color: #101218;
                border-top: 1px solid #1a202c;
                border-bottom: 1px solid #1c202a;
            }
        """)
        handle_layout = QHBoxLayout(self.tab_handle)
        handle_layout.setContentsMargins(0, 0, 0, 0)
        handle_layout.setSpacing(0)
        handle_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_collapse = QPushButton("⯆", self.tab_handle)
        self.btn_collapse.setObjectName("shelfCollapseBtn")
        self.btn_collapse.setFixedSize(36, 12)
        self.btn_collapse.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_collapse.setToolTip("Свернуть панель измерений")
        self.btn_collapse.setStyleSheet("""
            QPushButton#shelfCollapseBtn {
                background: transparent;
                color: #64748b;
                border: none;
                font-size: 10px;
                padding: 0;
                margin: 0;
            }
            QPushButton#shelfCollapseBtn:hover {
                color: #00e5ff;
            }
        """)
        self.btn_collapse.clicked.connect(self.toggle_collapsed)
        handle_layout.addWidget(self.btn_collapse)

        # Клик по самой полоске шторки также переключает состояние
        self.tab_handle.mousePressEvent = lambda e: self.toggle_collapsed() if e.button() == Qt.MouseButton.LeftButton else None
        root_vbox.addWidget(self.tab_handle)

        # ---------------- ТЕЛО ПАНЕЛИ (Shelf Body) ----------------
        self.shelf_body = QWidget(self)
        self.shelf_body.setObjectName("shelfBody")
        self.shelf_body.setFixedHeight(46)
        body_layout = QHBoxLayout(self.shelf_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Безопасный скролл: если окно сжимают, содержимое никогда не распирает окно!
        self.scroll_area = QScrollArea(self.shelf_body)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        inner_widget = QWidget()
        inner_widget.setObjectName("barInnerWidget")
        inner_widget.setStyleSheet("QWidget#barInnerWidget { background: transparent; }")
        bar_layout = QHBoxLayout(inner_widget)
        bar_layout.setContentsMargins(10, 0, 10, 0)
        bar_layout.setSpacing(8)
        bar_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # ---------------- ЗОНА 1: Компактный селектор режимов (иконки) ----------------
        self.mode_container = QFrame()
        self.mode_container.setObjectName("segmentedBox")
        self.mode_container.setStyleSheet("""
            QFrame#segmentedBox {
                background-color: #161920;
                border: 1px solid #2a3441;
                border-radius: 6px;
                padding: 2px;
            }
        """)
        mode_btn_layout = QHBoxLayout(self.mode_container)
        mode_btn_layout.setContentsMargins(2, 2, 2, 2)
        mode_btn_layout.setSpacing(2)

        self.btn_mode_ruler = QPushButton("📏")
        self.btn_mode_ruler.setFixedSize(28, 28)
        self.btn_mode_ruler.setCheckable(True)
        self.btn_mode_ruler.setChecked(True)
        self.btn_mode_ruler.setToolTip("<b>Линейка (M)</b><br>Линейный замер расстояний на небесной сфере.")

        self.btn_mode_disk = QPushButton("⭕")
        self.btn_mode_disk.setFixedSize(28, 28)
        self.btn_mode_disk.setCheckable(True)
        self.btn_mode_disk.setToolTip("<b>Привязка диска Луны</b><br>Растяните рамку 1:1 или укажите 3 точки на лимбе для сферической коррекции.")

        self.btn_mode_ellipse = QPushButton("⬭")
        self.btn_mode_ellipse.setFixedSize(28, 28)
        self.btn_mode_ellipse.setCheckable(True)
        self.btn_mode_ellipse.setToolTip("<b>Эллипс кратера</b><br>Замер истинного диаметра кратера по крупному плану.")

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_mode_ruler, 0)
        self.mode_group.addButton(self.btn_mode_disk, 1)
        self.mode_group.addButton(self.btn_mode_ellipse, 2)

        for btn in (self.btn_mode_ruler, self.btn_mode_disk, self.btn_mode_ellipse):
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #94a3b8;
                    border: none;
                    border-radius: 3px;
                    font-size: 13px;
                    font-weight: bold;
                    padding: 0;
                }
                QPushButton:hover {
                    color: #e2e8f0;
                    background-color: #1e2436;
                }
                QPushButton:checked {
                    background-color: #00adb5;
                    color: #111114;
                }
            """)
            mode_btn_layout.addWidget(btn)

        self.btn_mode_ruler.clicked.connect(lambda: self._on_mode_button_clicked("ruler"))
        self.btn_mode_disk.clicked.connect(lambda: self._on_mode_button_clicked("disk"))
        self.btn_mode_ellipse.clicked.connect(lambda: self._on_mode_button_clicked("ellipse"))

        bar_layout.addWidget(self.mode_container)

        # ---------------- ЗОНА 2: Контекстные кнопки и статус ----------------
        self.context_container = QWidget()
        ctx_layout = QHBoxLayout(self.context_container)
        ctx_layout.setContentsMargins(0, 0, 0, 0)
        ctx_layout.setSpacing(6)

        # 2.A Контекстный блок методов диска с кнопками подтверждения: [ Рамка ] [ 3 Точки ] / [ Метод ] [✓] [✕]
        self.disk_methods_widget = QFrame()
        self.disk_methods_widget.setObjectName("diskMethodsBox")
        self.disk_methods_widget.setStyleSheet("""
            QFrame#diskMethodsBox {
                background-color: #161920;
                border: 1px solid #2a3441;
                border-radius: 6px;
                padding: 2px;
            }
        """)
        method_layout = QHBoxLayout(self.disk_methods_widget)
        method_layout.setContentsMargins(2, 2, 2, 2)
        method_layout.setSpacing(2)

        self.btn_method_box = QPushButton("Рамка")
        self.btn_method_box.setCheckable(True)
        self.btn_method_box.setChecked(True)
        self.btn_method_box.setFixedHeight(22)
        self.btn_method_box.setToolTip("Построение диска растягиванием рамки 1:1 по лимбу с трансформацией")

        self.btn_method_3pts = QPushButton("3 Точки")
        self.btn_method_3pts.setCheckable(True)
        self.btn_method_3pts.setFixedHeight(22)
        self.btn_method_3pts.setToolTip("Построение диска по 3 кликам на видимой дуге лимба")

        self.method_btn_group = QButtonGroup(self)
        self.method_btn_group.setExclusive(True)
        self.method_btn_group.addButton(self.btn_method_box, 0)
        self.method_btn_group.addButton(self.btn_method_3pts, 1)

        for btn in (self.btn_method_box, self.btn_method_3pts):
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #94a3b8;
                    border: none;
                    border-radius: 3px;
                    font-size: 10px;
                    font-weight: bold;
                    padding: 1px 6px;
                }
                QPushButton:hover {
                    color: #e2e8f0;
                    background-color: #1e2436;
                }
                QPushButton:checked {
                    background-color: #00adb5;
                    color: #111114;
                }
            """)

        self.btn_apply_disk = QPushButton("✓")
        self.btn_apply_disk.setFixedSize(24, 24)
        self.btn_apply_disk.setEnabled(True)
        self.btn_apply_disk.setVisible(False)
        self.btn_apply_disk.setToolTip("Применить привязку диска (Enter)")
        self.btn_apply_disk.setStyleSheet("""
            QPushButton {
                background-color: #00adb5;
                color: #111114;
                font-weight: bold;
                font-size: 12px;
                border: none;
                border-radius: 4px;
                padding: 0;
            }
            QPushButton:hover:enabled {
                background-color: #00fff2;
            }
            QPushButton:disabled {
                background-color: #1e2433;
                color: #555e75;
            }
        """)
        self.btn_apply_disk.clicked.connect(self.apply_alignment_requested.emit)

        self.btn_cancel_disk = QPushButton("✕")
        self.btn_cancel_disk.setFixedSize(24, 24)
        self.btn_cancel_disk.setEnabled(True)
        self.btn_cancel_disk.setVisible(False)
        self.btn_cancel_disk.setToolTip("Сбросить построение (Esc)")
        self.btn_cancel_disk.setStyleSheet("""
            QPushButton {
                background-color: #242938;
                color: #cbd5e1;
                border: 1px solid #384259;
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover:enabled {
                background-color: #dc2626;
                color: #ffffff;
                border-color: #ef4444;
            }
            QPushButton:disabled {
                color: #555c70;
                background-color: #161822;
                border-color: #202430;
            }
        """)
        self.btn_cancel_disk.clicked.connect(self.reset_disk_requested.emit)

        method_layout.addWidget(self.btn_method_box)
        method_layout.addWidget(self.btn_method_3pts)
        method_layout.addWidget(self.btn_apply_disk)
        method_layout.addWidget(self.btn_cancel_disk)

        self.btn_method_box.clicked.connect(lambda: self.set_disk_method("center_radius"))
        self.btn_method_3pts.clicked.connect(lambda: self.set_disk_method("three_points"))
        ctx_layout.addWidget(self.disk_methods_widget)

        # 2.B Компактная плашка статуса откалиброванного диска (появляется ТОЛЬКО когда диск зафиксирован):
        # [ ● R: 121 px ]  [ ✕ Сброс ]  /  [ ● Calibrated (R: 121 px) ]  [ ✕ ]
        self.disk_status_box = QFrame()
        self.disk_status_box.setObjectName("diskStatusBox")
        self.disk_status_box.setStyleSheet("""
            QFrame#diskStatusBox {
                background-color: #161920;
                border: 1px solid #2a3441;
                border-radius: 6px;
                padding: 2px 6px;
            }
        """)
        status_hlayout = QHBoxLayout(self.disk_status_box)
        status_hlayout.setContentsMargins(6, 2, 6, 2)
        status_hlayout.setSpacing(6)

        self.lbl_disk_status = QLabel("")
        self.lbl_disk_status.setStyleSheet("color: #00e5ff; font-size: 11px; font-weight: bold; background: transparent; border: none; padding: 0 4px;")
        status_hlayout.addWidget(self.lbl_disk_status)

        self.btn_reset_disk = QPushButton("✕")
        self.btn_reset_disk.setFixedSize(24, 24)
        self.btn_reset_disk.setToolTip("Сбросить калибровку диска")
        self.btn_reset_disk.setStyleSheet("""
            QPushButton {
                background-color: #242938;
                color: #cbd5e1;
                border: 1px solid #384259;
                border-radius: 4px;
                font-size: 10px;
                padding: 1px 6px;
            }
            QPushButton:hover:enabled {
                background-color: #dc2626;
                color: #ffffff;
                border-color: #ef4444;
            }
        """)
        self.btn_reset_disk.clicked.connect(self.reset_disk_requested.emit)
        status_hlayout.addWidget(self.btn_reset_disk)
        self.disk_status_box.setVisible(False)
        ctx_layout.addWidget(self.disk_status_box)

        # 2.C Единый блок шага эллипса (для режима Ellipse, без вложенных рамок)
        self.ellipse_step_box = QFrame()
        self.ellipse_step_box.setObjectName("ellipseStepBox")
        self.ellipse_step_box.setStyleSheet("""
            QFrame#ellipseStepBox {
                background-color: #161920;
                border: 1px solid #2a3441;
                border-radius: 6px;
                padding: 2px 8px;
            }
        """)
        step_layout = QHBoxLayout(self.ellipse_step_box)
        step_layout.setContentsMargins(6, 2, 6, 2)
        step_layout.setSpacing(4)

        self.lbl_ellipse_step = QLabel("Шаг 1: тяните большую ось (2a)")
        self.lbl_ellipse_step.setStyleSheet("color: #00e5ff; font-size: 11px; font-weight: bold; background: transparent; border: none; padding: 0 4px;")
        self.lbl_ellipse_step.setToolTip("1) Тяните вдоль максимального поперечника (2a); 2) Растяните перпендикулярно для малой оси (2b)")
        step_layout.addWidget(self.lbl_ellipse_step)
        ctx_layout.addWidget(self.ellipse_step_box)

        bar_layout.addWidget(self.context_container)

        # Скрытая метка подсказки (для обратной совместимости вызовов)
        self.lbl_disk_prompt = QLabel("")
        self.lbl_disk_prompt.setVisible(False)

        # Служебные переключатели методов (для совместимости со старыми тестами)
        self.disk_options_box = self.context_container
        self.radio_center_radius = QRadioButton("Center + Radius")
        self.radio_center_radius.setChecked(True)
        self.radio_3points = QRadioButton("3 Points on Limb")
        self.method_group = QButtonGroup(self)
        self.method_group.addButton(self.radio_center_radius)
        self.method_group.addButton(self.radio_3points)
        self.radio_center_radius.toggled.connect(lambda c: self.set_disk_method("center_radius") if c else None)
        self.radio_3points.toggled.connect(lambda c: self.set_disk_method("three_points") if c else None)

        # Внутренний контейнер хранения метрик (всегда скрыт, данные рендерятся исключительно на холсте в бейдже)
        self.results_box = QFrame(self)
        self.results_box.hide()
        self.row_res_1 = QLabel("Proj: —", self.results_box)
        self.row_res_2 = QLabel("True: —", self.results_box)
        self.row_res_3 = QLabel("Tilt: —", self.results_box)

        bar_layout.addStretch()

        # Правый край: стильная компактная кнопка справки [ ? ]
        self.btn_help = QPushButton("?")
        self.btn_help.setFixedSize(24, 24)
        self.btn_help.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_help.setToolTip("Справочник: Сферическая проекция и физика измерений кратеров (F1)")
        self.btn_help.setStyleSheet("""
            QPushButton {
                background-color: #1e2230;
                color: #00e5ff;
                border: 1px solid #364156;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12px;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #00adb5;
                color: #111114;
                border-color: #00fff2;
            }
        """)
        self.btn_help.clicked.connect(self._show_help_dialog)
        bar_layout.addWidget(self.btn_help)

        # Псевдоним кнопки закрытия на шеврон шторки для совместимости
        self.btn_close = self.btn_collapse

        self.scroll_area.setWidget(inner_widget)
        body_layout.addWidget(self.scroll_area)
        root_vbox.addWidget(self.shelf_body)

        self.setStyleSheet("""
            QFrame#measurementBottomBar {
                background-color: #14171d;
                border-top: 1px solid #1a202c;
                border-bottom: none;
                border-left: none;
                border-right: none;
            }
            QLabel {
                background: transparent;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        # Применяем контекстную видимость для режима по умолчанию (ruler)
        self._update_contextual_visibility()

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 14 if self._is_collapsed else 60)

    @property
    def is_collapsed(self) -> bool:
        return self._is_collapsed

    @property
    def is_expanded(self) -> bool:
        return not self._is_collapsed

    @is_expanded.setter
    def is_expanded(self, val: bool):
        self.set_collapsed(not val)

    def toggle_collapsed(self):
        new_state = not self._is_collapsed
        self.set_collapsed(new_state)
        if new_state:
            self.close_requested.emit()

    def set_collapsed(self, collapsed: bool):
        new_val = bool(collapsed)
        if self._is_collapsed == new_val:
            return
        self._is_collapsed = new_val
        if self._is_collapsed:
            self.shelf_body.setVisible(False)
            self.setFixedHeight(14)
            self.btn_collapse.setText("⯅")
            self.btn_collapse.setToolTip("Развернуть панель измерений")
            self.btn_close.setText("⯅")
            self.tab_handle.setStyleSheet("""
                QFrame#shelfTabHandle {
                    background-color: #101218;
                    border-top: 2px solid #00e5ff;
                    border-bottom: none;
                }
            """)
        else:
            self.shelf_body.setVisible(True)
            self.setFixedHeight(60)
            self.btn_collapse.setText("⯆")
            self.btn_collapse.setToolTip("Свернуть панель измерений")
            self.btn_close.setText("⯆")
            self.tab_handle.setStyleSheet("""
                QFrame#shelfTabHandle {
                    background-color: #101218;
                    border-top: 1px solid #1a202c;
                    border-bottom: 1px solid #1c202a;
                }
            """)

    def _show_help_dialog(self):
        dlg = MeasurementHelpDialog(self)
        dlg.exec()

    def _on_mode_button_clicked(self, mode: str):
        self._current_mode = mode
        self._update_contextual_visibility()
        self.mode_changed.emit(mode)

    def set_disk_method(self, method: str):
        if getattr(self, '_setting_disk_method', False):
            return
        self._setting_disk_method = True
        try:
            self._disk_method = method
            if method == "three_points":
                self.btn_method_3pts.setChecked(True)
                self.radio_3points.blockSignals(True)
                self.radio_3points.setChecked(True)
                self.radio_3points.blockSignals(False)
            else:
                self.btn_method_box.setChecked(True)
                self.radio_center_radius.blockSignals(True)
                self.radio_center_radius.setChecked(True)
                self.radio_center_radius.blockSignals(False)
            self._update_disk_buttons()
            self.disk_method_changed.emit(method)
        finally:
            self._setting_disk_method = False

    def _update_disk_buttons(self):
        if self._is_calibrated:
            self.disk_methods_widget.setVisible(False)
            self.btn_apply_disk.setVisible(False)
            self.btn_cancel_disk.setVisible(False)
            self.disk_status_box.setVisible(self._current_mode in ("disk", "ruler"))
            if self._current_mode == "disk":
                self.lbl_disk_status.setText(f"● R: {int(round(self._calibrated_radius))} px")
                self.lbl_disk_status.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 11px; background: transparent; border: none; padding: 0 4px;")
                self.btn_reset_disk.setText("✕ Сброс")
                self.btn_reset_disk.setFixedSize(56, 22)
            else:
                self.lbl_disk_status.setText(f"● Calibrated (R: {int(round(self._calibrated_radius))} px)")
                self.lbl_disk_status.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 11px; background: transparent; border: none; padding: 0 4px;")
                self.btn_reset_disk.setText("✕")
                self.btn_reset_disk.setFixedSize(24, 22)
            self.btn_reset_disk.setEnabled(True)
        else:
            self.disk_status_box.setVisible(False)
            self.btn_reset_disk.setEnabled(False)
            self.lbl_disk_status.setText("")
            if self._current_mode == "disk":
                self.disk_methods_widget.setVisible(True)
                if getattr(self, '_pending_active', False):
                    if self._disk_method == "center_radius":
                        self.btn_method_box.setVisible(True)
                        self.btn_method_3pts.setVisible(False)
                    else:
                        self.btn_method_box.setVisible(False)
                        self.btn_method_3pts.setVisible(True)
                    self.btn_apply_disk.setVisible(True)
                    self.btn_cancel_disk.setVisible(True)
                else:
                    self.btn_method_box.setVisible(True)
                    self.btn_method_3pts.setVisible(True)
                    self.btn_apply_disk.setVisible(False)
                    self.btn_cancel_disk.setVisible(False)
            else:
                self.disk_methods_widget.setVisible(False)
                self.btn_apply_disk.setVisible(False)
                self.btn_cancel_disk.setVisible(False)

    def _update_contextual_visibility(self):
        self._update_disk_buttons()
        self.ellipse_step_box.setVisible(self._current_mode == "ellipse")

    def set_mode(self, mode: str):
        self._current_mode = mode
        if mode == "ruler":
            self.btn_mode_ruler.setChecked(True)
        elif mode == "disk":
            self.btn_mode_disk.setChecked(True)
        elif mode == "ellipse":
            self.btn_mode_ellipse.setChecked(True)
        self._update_contextual_visibility()

    def set_disk_status(self, calibrated: bool, radius_px: float = 0.0):
        self._is_calibrated = calibrated
        self._calibrated_radius = radius_px
        if calibrated:
            self._pending_active = False
        self._update_disk_buttons()

    def set_disk_prompt(self, text: str, can_apply: bool = False):
        self.lbl_disk_prompt.setText(text)
        self._pending_active = can_apply
        self.btn_apply_disk.setEnabled(can_apply)
        self._update_disk_buttons()

    def set_ellipse_step(self, step: int):
        base_style = "font-size: 11px; font-weight: bold; background: transparent; border: none; padding: 0 4px;"
        if step == 1:
            self.lbl_ellipse_step.setText("Шаг 1: тяните большую ось (2a)")
            self.lbl_ellipse_step.setStyleSheet(f"color: #00e5ff; {base_style}")
        elif step == 2:
            self.lbl_ellipse_step.setText("Шаг 2: форма вала кратера (2b)")
            self.lbl_ellipse_step.setStyleSheet(f"color: #38bdf8; {base_style}")
        elif step == 3:
            self.lbl_ellipse_step.setText("Кратер зафиксирован (клик для сброса)")
            self.lbl_ellipse_step.setStyleSheet(f"color: #4ade80; {base_style}")
        else:
            self.lbl_ellipse_step.setText("Шаг 1: тяните большую ось (2a)")
            self.lbl_ellipse_step.setStyleSheet(f"color: #94a3b8; {base_style}")

    def set_ruler_results(
        self,
        projected_km: float,
        ang_str: str,
        dist_px: float,
        true_km: Optional[float] = None,
        tilt_deg: float = 0.0,
        cos_theta: float = 1.0,
        is_calibrated: bool = False
    ):
        self._last_proj_km = projected_km
        self._last_true_km = true_km
        self._last_tilt_deg = tilt_deg
        self._last_cos_theta = cos_theta
        self.row_res_1.setText(f"Proj: {projected_km:.1f} km ({dist_px:.0f} px)")
        if is_calibrated and true_km is not None:
            self.row_res_2.setText(f"True: {true_km:.1f} km")
            self.row_res_3.setText(f"Tilt: {tilt_deg:.1f}°")
        else:
            self.row_res_2.setText("True: —")
            self.row_res_3.setText("Tilt: —")

    def set_ellipse_results(
        self,
        true_km: float,
        apparent_km: float,
        tilt_deg: float,
        cos_theta: float,
        eccentricity: float,
        semi_major_px: float,
        semi_minor_px: float
    ):
        self.row_res_1.setText(f"True Dia: {true_km:.1f} km")
        self.row_res_2.setText(f"Apparent: {apparent_km:.1f} km")
        self.row_res_3.setText(f"Tilt: {tilt_deg:.0f}° (cos θ = {cos_theta:.2f})")

    def clear_results(self):
        self.row_res_1.setText("Proj: —")
        self.row_res_2.setText("True: —")
        self.row_res_3.setText("Tilt: —")


# Псевдоним для полной обратной совместимости
MeasurementHUD = MeasurementBottomBar
