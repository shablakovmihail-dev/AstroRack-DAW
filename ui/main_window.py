#"""Главное окно приложения AstroRackWindow с DAW-тулбаром и конвейером обработки."""
import os
import time
import json
import traceback
import numpy as np
from scipy.ndimage import map_coordinates

from PyQt6.QtCore import (
    Qt, QThread, QPointF, QRectF, QTimer, pyqtProperty,
    QPropertyAnimation, QEasingCurve, QSize, QSignalBlocker
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFrame, QSplitter, QComboBox, QMessageBox, QScrollArea,
    QSizePolicy, QPushButton, QLabel, QFileDialog, QDialog
)
from PyQt6.QtGui import (
    QKeySequence, QShortcut, QDragEnterEvent, QDropEvent, QColor, QBrush, QPalette,
    QIcon, QPixmap
)

from utils.file_io import load_image, save_image, get_resource_path
from core.history import HistoryManager
from core.processing import AstroEngine
from core.pipeline import PipelineWorker
from core.sound import sound_engine
from .canvas import AstroCanvas
from .monitors import MiniHistogramWidget, ProfileMonitor
from .widgets import (
    WaveletInsert, RichardsonLucyInsert, ToneContrastInsert, ColorMonoInsert,
    LocalContrastInsert, ColorEngineInsert,
    ToolboxPanel, PresetComboDelegate, PsfInspectorDialog, TelescopeOpticsDialog,
    THEMES, StateCopyButton, AstroHistogramHeader
)
from .dialogs import AboutDialog, ResetConfirmDialog


class CollapsibleRackContainer(QWidget):
    """Контейнер боковой панели инсертов с поддержкой полного сжатия в QSplitter."""

    def minimumSizeHint(self) -> QSize:
        return QSize(16, 0)


class RackInnerWidget(QWidget):
    """Внутренний контейнер панели инсертов с нулевым минимальным размером для анимации."""

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 0)


class StatusProxy:
    """Прокси для перенаправления вызовов setText статуса на плавающий оверлей холста и статус-бар."""

    def __init__(self, canvas, status_label=None):
        self._canvas = canvas
        self._status_label = status_label

    def setText(self, text: str):
        self._canvas.set_status_text(text)
        if self._status_label is not None:
            self._status_label.setText(str(text))

    def text(self) -> str:
        return self._canvas.status_overlay.text()


class AstroRackWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AstroRack DAW v7.0 — Lunar & Planetary Processor")
        self.resize(1500, 930)
        self.setAcceptDrops(True)

        self._current_processed_image = None
        self._raw_image = None
        self._current_filename = ""

        default_dir = os.path.expanduser("~")
        self._last_open_dir = default_dir
        self._last_save_dir = default_dir

        self._state_a = None
        self._state_b = None
        self._active_ab = 'A'

        self._profile_dialog = None
        self._psf_dialog = None
        self._optics_dialog = None
        self.history = HistoryManager(max_depth=50)
        self._last_dispatched_params = None
        self._render_start_time = 0.0

        # Троттлинг/дебаунс превью-рендера при движении ползунков (~35 мс)
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(35)
        self._debounce_timer.timeout.connect(self._on_debounce_timeout)

        # Таймер финального чистового рендера после остановки ползунка (~110 мс)
        self._final_render_timer = QTimer(self)
        self._final_render_timer.setSingleShot(True)
        self._final_render_timer.setInterval(110)
        self._final_render_timer.timeout.connect(lambda: self._execute_render(preview=False))

        # Троттлинг обновления гистограммы, чтобы не задерживать отрисовку холста
        self._hist_timer = QTimer(self)
        self._hist_timer.setSingleShot(True)
        self._hist_timer.setInterval(50)
        self._hist_timer.timeout.connect(self._flush_histogram_update)

        self._builtin_presets = {
            "Flat State (All Bypassed)": {
                "insert1_enabled": False, "w_mode": 0, "w_l1_on": False, "w_l2_on": False, "w_denoise_on": False,
                "w_l1": 1.00, "w_l2": 1.00, "w_denoise": 0.0, "w_wet_mix": 100.0,
                "insert2_enabled": False, "rl_sigma": 1.0, "rl_iters": 10, "rl_anti_ringing": False, "rl_wet_mix": 100.0,
                "insert3_enabled": False, "tc_black_on": False, "tc_gamma_on": False, "tc_contrast_on": False,
                "tc_black": 0.000, "tc_gamma": 1.00, "tc_contrast": 1.00,
                "insert4_enabled": False, "auto_rgb_align": False, "extract_green": False,
                "insert5_enabled": False, "lc_method": "clahe", "lc_amount": 1.00, "lc_radius": 16.0, "lc_wet_mix": 100.0,
                "insert6_enabled": False, "cs_saturation": 1.00, "cs_mineral": 0.00, "cs_bg_neutralize": False, "cs_wet_mix": 100.0,
            },
            "Moon: Fine Craters Sharp": {
                "insert1_enabled": True, "w_mode": 0, "w_l1_on": True, "w_l2_on": True, "w_denoise_on": True,
                "w_l1": 2.50, "w_l2": 1.40, "w_denoise": 0.05, "w_wet_mix": 100.0,
                "insert2_enabled": True, "rl_sigma": 1.1, "rl_iters": 8, "rl_anti_ringing": True, "rl_wet_mix": 100.0,
                "insert3_enabled": True, "tc_black_on": True, "tc_gamma_on": True, "tc_contrast_on": True,
                "tc_black": 0.015, "tc_gamma": 1.05, "tc_contrast": 1.20,
                "insert4_enabled": True, "auto_rgb_align": True, "extract_green": False,
                "insert5_enabled": True, "lc_method": "clahe", "lc_amount": 1.80, "lc_radius": 16.0, "lc_wet_mix": 100.0,
                "insert6_enabled": False, "cs_saturation": 1.00, "cs_mineral": 0.00, "cs_bg_neutralize": False, "cs_wet_mix": 100.0,
            },
            "Moon: Ultra-Resolution Peak (7.0x)": {
                "insert1_enabled": True, "w_mode": 0, "w_l1_on": True, "w_l2_on": True, "w_denoise_on": True,
                "w_l1": 4.80, "w_l2": 2.20, "w_denoise": 0.12, "w_wet_mix": 100.0,
                "insert2_enabled": False, "rl_sigma": 1.2, "rl_iters": 10, "rl_anti_ringing": True, "rl_wet_mix": 100.0,
                "insert3_enabled": True, "tc_black_on": True, "tc_gamma_on": True, "tc_contrast_on": True,
                "tc_black": 0.035, "tc_gamma": 0.90, "tc_contrast": 1.45,
                "insert4_enabled": False, "auto_rgb_align": False, "extract_green": False,
                "insert5_enabled": True, "lc_method": "clahe", "lc_amount": 2.40, "lc_radius": 24.0, "lc_wet_mix": 100.0,
                "insert6_enabled": False, "cs_saturation": 1.00, "cs_mineral": 0.00, "cs_bg_neutralize": False, "cs_wet_mix": 100.0,
            },
            "Planetary: Lucy-Richardson Focus": {
                "insert1_enabled": True, "w_mode": 0, "w_l1_on": True, "w_l2_on": True, "w_denoise_on": True,
                "w_l1": 1.60, "w_l2": 1.25, "w_denoise": 0.03, "w_wet_mix": 100.0,
                "insert2_enabled": True, "rl_sigma": 1.3, "rl_iters": 16, "rl_anti_ringing": True, "rl_wet_mix": 100.0,
                "insert3_enabled": True, "tc_black_on": True, "tc_gamma_on": True, "tc_contrast_on": True,
                "tc_black": 0.010, "tc_gamma": 1.00, "tc_contrast": 1.15,
                "insert4_enabled": True, "auto_rgb_align": True, "extract_green": False,
                "insert5_enabled": False, "lc_method": "high_pass", "lc_amount": 0.80, "lc_radius": 30.0, "lc_wet_mix": 100.0,
                "insert6_enabled": True, "cs_saturation": 1.20, "cs_mineral": 0.00, "cs_bg_neutralize": False, "cs_wet_mix": 100.0,
            },
            "Moon: Mineral Color & Clarity": {
                "insert1_enabled": True, "w_mode": 0, "w_l1_on": True, "w_l2_on": True, "w_denoise_on": True,
                "w_l1": 2.20, "w_l2": 1.30, "w_denoise": 0.04, "w_wet_mix": 100.0,
                "insert2_enabled": False, "rl_sigma": 1.1, "rl_iters": 8, "rl_anti_ringing": True, "rl_wet_mix": 100.0,
                "insert3_enabled": True, "tc_black_on": True, "tc_gamma_on": True, "tc_contrast_on": True,
                "tc_black": 0.015, "tc_gamma": 1.00, "tc_contrast": 1.15,
                "insert4_enabled": True, "auto_rgb_align": True, "extract_green": False,
                "insert5_enabled": True, "lc_method": "clahe", "lc_amount": 2.20, "lc_radius": 16.0, "lc_wet_mix": 100.0,
                "insert6_enabled": True, "cs_saturation": 1.60, "cs_mineral": 0.85, "cs_bg_neutralize": True, "cs_wet_mix": 100.0,
            }
        }

        self.thread = QThread()
        self.worker = PipelineWorker()
        self.worker.moveToThread(self.thread)
        self.worker.frame_ready.connect(self._on_frame_rendered)
        self.worker.rgb_shifts_computed.connect(self._on_rgb_shifts_computed)
        self.thread.start()

        self._panel_collapsed = False
        self._last_panel_width = 380

        self._setup_ui()
        self._apply_dark_theme()
        self.initUI()

        self._panel_anim = QPropertyAnimation(self, b"panelWidth")
        self._panel_anim.setDuration(220)
        self._panel_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._panel_anim.finished.connect(self._on_panel_anim_finished)

        self.canvas.zoomChanged.connect(lambda act: self._record_current_state(act))
        self.toolbox.loupe_zoom_changed.connect(self.canvas.set_loupe_zoom)
        self._setup_shortcuts()

        self._reset_all_to_flat()
        self._record_current_state("Initial State")
        self._state_a = self._get_current_state_dict()
        self._state_b = self._get_current_state_dict()
        self._update_canvas_guard_state()

    def initUI(self):
        """Установка иконки окна и базовых параметров интерфейса."""
        ico_path = get_resource_path(os.path.join("assets", "app_icon.ico"))
        icon_256_path = get_resource_path(os.path.join("assets", "icons", "icon_256x256.png"))
        if os.path.isfile(icon_256_path):
            self.setWindowIcon(QIcon(icon_256_path))
        elif os.path.isfile(ico_path):
            self.setWindowIcon(QIcon(ico_path))
        else:
            self.setWindowIcon(QIcon(get_resource_path(os.path.join("assets", "icons", "icon_256x256.png"))))

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_vbox = QVBoxLayout(central_widget)
        root_vbox.setContentsMargins(8, 8, 8, 0)
        root_vbox.setSpacing(8)

        # 1. ПЕРВООЧЕРЕДНАЯ ИНИЦИАЛИЗАЦИЯ CANVAS
        self.canvas = AstroCanvas(self)
        self.canvas.cropApplied.connect(self._on_crop_applied)
        self.canvas.cropCancelled.connect(self._on_crop_cancelled)
        self.canvas.profileLineSelected.connect(self._on_profile_line_selected)
        self.canvas.fileDropped.connect(self.open_image)
        self.canvas.openFileDialogRequested.connect(self._open_file_dialog)

        # 2. ВЕРХНЯЯ ПАНЕЛЬ (DAW TOOLBAR) - растянута на всю ширину окна
        top_bar_widget = QWidget()
        top_bar_widget.setFixedHeight(48)
        top_bar = QHBoxLayout(top_bar_widget)
        top_bar.setContentsMargins(0, 6, 8, 6)
        top_bar.setSpacing(6)

        def create_v_divider():
            line = QFrame()
            line.setFrameShape(QFrame.Shape.VLine)
            line.setFrameShadow(QFrame.Shadow.Plain)
            line.setStyleSheet("color: #2c2c38; background-color: #2c2c38; width: 1px; margin: 4px 0px;")
            return line

        # BRAND LOGO (Ширина контейнера СТРОГО 48px = ширине вертикального левого тулбара)
        self.logo_container = QWidget()
        self.logo_container.setFixedWidth(48)
        self.logo_container.setFixedHeight(36)
        logo_layout = QHBoxLayout(self.logo_container)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(0)
        logo_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_logo = QLabel()
        self.lbl_logo.setFixedSize(32, 32)
        self.lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_logo.setStyleSheet("margin: 0; padding: 0; background: transparent;")
        self.lbl_logo.setToolTip("AstroRack DAW v7.0")
        icon_32_path = get_resource_path(os.path.join("assets", "icons", "icon_32x32.png"))
        if os.path.isfile(icon_32_path):
            self.lbl_logo.setPixmap(QPixmap(icon_32_path).scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            icon_24_path = get_resource_path(os.path.join("assets", "icons", "icon_24x24.png"))
            if os.path.isfile(icon_24_path):
                self.lbl_logo.setPixmap(QPixmap(icon_24_path).scaled(28, 28, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        logo_layout.addWidget(self.lbl_logo)
        top_bar.addWidget(self.logo_container)

        # FILE (OPEN & EXPORT & RESET) - БЕЗ разделителя между иконкой и кнопкой импорта, spacing=6px!
        self.btn_open = QPushButton("⇩")
        self.btn_open.setFixedSize(36, 36)
        self.btn_open.setObjectName("toolIconBtn")
        self.btn_open.setToolTip("Open Image (Ctrl+O)")
        self.btn_open.clicked.connect(self._open_file_dialog)

        self.btn_export = QPushButton("⇪")
        self.btn_export.setFixedSize(36, 36)
        self.btn_export.setObjectName("toolIconBtn")
        self.btn_export.setToolTip("Export Image (Ctrl+S)")
        self.btn_export.clicked.connect(self._export_image)

        self.btn_reset_image = QPushButton("✕")
        self.btn_reset_image.setFixedSize(36, 36)
        self.btn_reset_image.setObjectName("btnResetImage")
        self.btn_reset_image.setToolTip("Сбросить снимок и вернуться на начальный экран (✕)")
        self.btn_reset_image.clicked.connect(self._on_reset_image_clicked)

        top_bar.addWidget(self.btn_open)
        top_bar.addWidget(self.btn_export)
        top_bar.addWidget(self.btn_reset_image)
        top_bar.addSpacing(6)
        top_bar.addWidget(create_v_divider())
        top_bar.addSpacing(6)

        # VIEW & SPLIT
        self.btn_split_view = QPushButton("Split View")
        self.btn_split_view.setCheckable(True)
        self.btn_split_view.setFixedHeight(36)
        self.btn_split_view.toggled.connect(self._on_split_toggled)
        self.btn_split_view.setToolTip(
            "<b>Раздельный просмотр (Split View) [S]</b><br>"
            "Интерактивное сравнение до/после обработки.<br><br>"
            "• <b>ЛКМ + перетаскивание линии</b> — сдвиг границы раздела.<br>"
            "• <b>[ ⇄ ]</b> — смена сторон (Before ↔ After).<br>"
            "• <b>[ ⇅ ]</b> — переключение горизонтального / вертикального разделения.<br>"
            "• <i>Фаза Луны на маркере указывает активную сторону After.</i>"
        )

        self.btn_invert_split = QPushButton("⇄")
        self.btn_invert_split.setCheckable(True)
        self.btn_invert_split.setFixedSize(36, 36)
        self.btn_invert_split.setObjectName("toolIconBtn")
        self.btn_invert_split.setToolTip("Инвертировать стороны До/После (⇄)")
        self.btn_invert_split.setVisible(False)
        self.btn_invert_split.toggled.connect(self._on_invert_split_toggled)

        self.btn_split_orient = QPushButton("⇅")
        self.btn_split_orient.setCheckable(True)
        self.btn_split_orient.setFixedSize(36, 36)
        self.btn_split_orient.setObjectName("toolIconBtn")
        self.btn_split_orient.setToolTip("Переключить сплит: Вертикальный / Горизонтальный (⇅)")
        self.btn_split_orient.setVisible(False)
        self.btn_split_orient.toggled.connect(self._on_split_orient_toggled)

        top_bar.addWidget(self.btn_split_view)
        top_bar.addWidget(self.btn_invert_split)
        top_bar.addWidget(self.btn_split_orient)
        top_bar.addSpacing(6)
        top_bar.addWidget(create_v_divider())
        top_bar.addSpacing(6)

        # PRESETS & HOTKEYS
        lbl_preset = QLabel("Preset:")
        lbl_preset.setStyleSheet("color: #888898; font-weight: bold; font-size: 11px; margin-left: 2px;")

        self.combo_presets = QComboBox()
        self.combo_presets.setFixedHeight(36)
        self.combo_presets.setFixedWidth(230)
        self.combo_presets.setItemDelegate(PresetComboDelegate(self.combo_presets))
        self._last_preset_index = 0
        self._populate_presets_combo()
        self.combo_presets.currentIndexChanged.connect(self._on_preset_selected)

        top_bar.addWidget(lbl_preset)
        top_bar.addWidget(self.combo_presets)

        # RIGHT BLOCK (ABOUT, HOTKEYS & HISTORY UNDO/REDO)
        top_bar.addStretch()

        # Миниатюрный тумблер тактильного звука UI (Subtle Audio Engine)
        self.btn_sound = QPushButton("🔊")
        self.btn_sound.setFixedSize(18, 18)
        self.btn_sound.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sound.clicked.connect(self._toggle_sound_effects)
        top_bar.addWidget(self.btn_sound)
        self._update_sound_button_ui(sound_engine.is_enabled())
        top_bar.addSpacing(6)

        self.btn_about = QPushButton("?")
        self.btn_about.setFixedSize(36, 36)
        self.btn_about.setObjectName("toolIconBtn")
        self.btn_about.setToolTip("О программе (About AstroRack)")
        self.btn_about.setStyleSheet("""
            QPushButton#toolIconBtn {
                background-color: #1c1c24;
                border: 1px solid #2c2c38;
                border-radius: 5px;
                color: #00adb5;
                font-size: 15px;
                font-weight: bold;
                font-family: 'Segoe UI', Arial, sans-serif;
                padding: 0;
                text-align: center;
            }
            QPushButton#toolIconBtn:hover {
                background-color: #262632;
                border-color: #00fff2;
                color: #ffffff;
            }
            QPushButton#toolIconBtn:pressed {
                background-color: #00adb5;
                color: #111114;
            }
        """)
        self.btn_about.clicked.connect(self._show_about_dialog)
        top_bar.addWidget(self.btn_about)

        self.btn_hotkeys = QPushButton("⌨")
        self.btn_hotkeys.setFixedSize(36, 36)
        self.btn_hotkeys.setObjectName("toolIconBtn")
        self.btn_hotkeys.setToolTip("Hotkeys Reference (F1)")
        self.btn_hotkeys.setStyleSheet("""
            QPushButton#toolIconBtn {
                background-color: #1c1c24;
                border: 1px solid #2c2c38;
                border-radius: 5px;
                color: #c8c8d4;
                font-size: 16px;
                font-family: 'Segoe UI Symbol', 'Segoe UI', Arial, sans-serif;
                padding: 0;
                text-align: center;
            }
            QPushButton#toolIconBtn:hover {
                background-color: #262632;
                border-color: #00adb5;
                color: #ffffff;
            }
            QPushButton#toolIconBtn:pressed {
                background-color: #00adb5;
                color: #111114;
            }
        """)
        self.btn_hotkeys.clicked.connect(self._show_hotkeys_dialog)
        top_bar.addWidget(self.btn_hotkeys)

        top_bar.addSpacing(6)
        top_bar.addWidget(create_v_divider())
        top_bar.addSpacing(6)

        self.btn_undo = QPushButton("⟲")
        self.btn_undo.setFixedSize(36, 36)
        self.btn_undo.setObjectName("toolIconBtn")
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_undo.clicked.connect(self._on_undo_clicked)

        self.btn_redo = QPushButton("⟳")
        self.btn_redo.setFixedSize(36, 36)
        self.btn_redo.setObjectName("toolIconBtn")
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")
        self.btn_redo.clicked.connect(self._on_redo_clicked)

        top_bar.addWidget(self.btn_undo)
        top_bar.addWidget(self.btn_redo)

        root_vbox.addWidget(top_bar_widget)

        # 3. РАБОЧАЯ ОБЛАСТЬ: ЛЕВАЯ ПАНЕЛЬ ИНСТРУМЕНТОВ + СПЛИТТЕР (ХОЛСТ И РЭК)
        work_hbox = QHBoxLayout()
        work_hbox.setContentsMargins(0, 0, 0, 0)
        work_hbox.setSpacing(6)

        # ЛЕВАЯ ПАНЕЛЬ ИНСТРУМЕНТОВ (TOOLBOX)
        self.toolbox = ToolboxPanel(self)
        self.toolbox.tool_changed.connect(self._on_toolbox_tool_changed)
        self.toolbox.rotate_cw.connect(lambda: self._apply_geometry(AstroEngine.rotate_90_cw, "Rotate 90° CW"))
        self.toolbox.rotate_ccw.connect(lambda: self._apply_geometry(AstroEngine.rotate_90_ccw, "Rotate 90° CCW"))
        self.toolbox.flip_h.connect(lambda: self._apply_geometry(AstroEngine.flip_horizontal, "Flip Horizontal"))
        self.toolbox.flip_v.connect(lambda: self._apply_geometry(AstroEngine.flip_vertical, "Flip Vertical"))
        self.toolbox.apply_crop.connect(self.canvas.apply_current_crop)
        self.toolbox.ruler_optics_requested.connect(self._show_optics_dialog)
        self.toolbox.ruler_optics_changed.connect(self.canvas.set_telescope_params)
        self.toolbox.ruler_unit_changed.connect(self.canvas.set_ruler_unit)
        self.canvas.psfPointClicked.connect(self._on_psf_point_clicked)
        self.canvas.rulerMeasured.connect(self._on_ruler_measured)
        work_hbox.addWidget(self.toolbox)

        # 4. РАБОЧАЯ ОБЛАСТЬ: SPLITTER И ПАНЕЛЬ ИНСЕРТОВ
        # Правая панель (Insert Rack): тянущаяся через QSplitter со складной кнопкой слева
        self.rack_container = CollapsibleRackContainer()
        self.rack_container.setMinimumWidth(340)
        self.rack_container.setMaximumWidth(700)
        rack_outer_hbox = QHBoxLayout(self.rack_container)
        rack_outer_hbox.setContentsMargins(0, 0, 0, 0)
        rack_outer_hbox.setSpacing(0)

        # Контейнер для кнопки сворачивания: центрирует кнопку по вертикали, предотвращая растягивание в полосу
        toggle_bar = QWidget()
        toggle_bar.setFixedWidth(16)
        toggle_bar.setStyleSheet("background-color: transparent;")
        toggle_vbox = QVBoxLayout(toggle_bar)
        toggle_vbox.setContentsMargins(0, 0, 0, 0)
        toggle_vbox.setSpacing(0)
        toggle_vbox.addStretch()

        self.btn_panel_toggle = QPushButton("►")
        self.btn_panel_toggle.setFixedSize(16, 52)
        self.btn_panel_toggle.setToolTip("Свернуть панель настроек")
        self.btn_panel_toggle.setStyleSheet("""
            QPushButton {
                background-color: #1e1e28;
                border: 1px solid #363648;
                border-radius: 5px;
                color: #00adb5;
                font-size: 10px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #00adb5;
                border-color: #00fff2;
                color: #111114;
            }
        """)
        self.btn_panel_toggle.clicked.connect(self._toggle_right_panel)
        toggle_vbox.addWidget(self.btn_panel_toggle)
        toggle_vbox.addStretch()

        rack_outer_hbox.addWidget(toggle_bar)

        # Внутреннее содержимое панели инсертов
        self.rack_inner_widget = RackInnerWidget()
        rack_vbox = QVBoxLayout(self.rack_inner_widget)
        rack_vbox.setContentsMargins(8, 0, 8, 0)
        rack_vbox.setSpacing(8)

        self.hist_header = AstroHistogramHeader()
        self.histogram = MiniHistogramWidget()
        self.hist_header.attach_histogram(self.histogram)
        rack_vbox.addWidget(self.hist_header)
        rack_vbox.addWidget(self.histogram)

        # БЛОК СРАВНЕНИЯ STATE A / B (НЕПОСРЕДСТВЕННО НАД MASTER BYPASS)
        ab_widget = QWidget()
        ab_hbox = QHBoxLayout(ab_widget)
        ab_hbox.setContentsMargins(0, 0, 0, 0)
        ab_hbox.setSpacing(6)

        self.btn_state_toggle = QPushButton("State: [ A ]")
        self.btn_state_toggle.setFixedHeight(24)
        self.btn_state_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_state_toggle.clicked.connect(self._toggle_ab_state)
        self._update_state_toggle_ui()

        # Совместимость с внутренними ссылками
        self.btn_state_a = self.btn_state_toggle
        self.btn_state_b = self.btn_state_toggle

        self.btn_copy_ab = StateCopyButton("A", "B")
        self.btn_copy_ab.setToolTip("Скопировать настройки из State A в State B")
        self.btn_copy_ab.clicked.connect(self._copy_a_to_b)

        self.btn_copy_ba = StateCopyButton("B", "A")
        self.btn_copy_ba.setToolTip("Скопировать настройки из State B в State A")
        self.btn_copy_ba.clicked.connect(self._copy_b_to_a)

        ab_hbox.addWidget(self.btn_state_toggle, stretch=1)

        sep_ab = QFrame()
        sep_ab.setFrameShape(QFrame.Shape.VLine)
        sep_ab.setStyleSheet("background-color: #363648; width: 1px; margin: 2px 2px;")
        ab_hbox.addWidget(sep_ab)

        ab_hbox.addWidget(self.btn_copy_ab)
        ab_hbox.addWidget(self.btn_copy_ba)
        rack_vbox.addWidget(ab_widget)

        self.btn_global_bypass = QPushButton("MASTER BYPASS: INACTIVE (EFFECTS ON)")
        self.btn_global_bypass.setCheckable(True)
        self.btn_global_bypass.setFixedHeight(28)
        self.btn_global_bypass.setStyleSheet("""
            QPushButton {
                background-color: #1e1e26;
                border: 1px solid #363646;
                border-radius: 5px;
                color: #00adb5;
                font-weight: bold;
                font-size: 11px;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                border-color: #00fff2;
            }
            QPushButton:checked {
                background-color: #ff3344;
                border-color: #ff6677;
                color: #ffffff;
            }
        """)
        self.btn_global_bypass.toggled.connect(self._on_global_bypass_toggled)
        rack_vbox.addWidget(self.btn_global_bypass)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 4, 0, 0)
        scroll_layout.setSpacing(8)

        rack_header = QLabel("EFFECT INSERTS RACK")
        rack_header.setStyleSheet("color: #555562; font-weight: bold; font-size: 10px; letter-spacing: 1px;")
        scroll_layout.addWidget(rack_header)

        self.insert1_wavelet = WaveletInsert()
        self.insert2_rl = RichardsonLucyInsert()
        self.insert3_tone = ToneContrastInsert()
        self.insert4_mono = ColorMonoInsert()
        self.insert5_lc = LocalContrastInsert()
        self.insert6_color = ColorEngineInsert()

        for insert in [
            self.insert1_wavelet, self.insert2_rl, self.insert3_tone,
            self.insert4_mono, self.insert5_lc, self.insert6_color
        ]:
            insert.slider_changed.connect(self.schedule_render)
            insert.bypass_toggled.connect(self.trigger_render_immediate)
            insert.state_committed.connect(self._on_parameter_committed)

        # Подключение сигналов изменения Dry/Wet в шапках инсертов
        self.insert1_wavelet.dry_wet_changed.connect(
            lambda val: self.update_insert_param(1, 'wet_mix', val / 100.0)
        )
        self.insert2_rl.dry_wet_changed.connect(
            lambda val: self.update_insert_param(2, 'wet_mix', val / 100.0)
        )
        self.insert5_lc.dry_wet_changed.connect(
            lambda val: self.update_insert_param(5, 'wet_mix', val / 100.0)
        )
        self.insert6_color.dry_wet_changed.connect(
            lambda val: self.update_insert_param(6, 'wet_mix', val / 100.0)
        )

        self.insert1_wavelet.layers_changed.connect(self._on_wavelet_layers_changed)

        # Синхронизация визиров гистограммы с ползунками Insert 3 (Tone & Contrast)
        self.insert3_tone.slider_changed.connect(self._sync_histogram_visors)
        self.insert3_tone.bypass_toggled.connect(self._sync_histogram_visors)
        self.insert3_tone.state_committed.connect(self._sync_histogram_visors)

        scroll_layout.addWidget(self.insert1_wavelet)
        scroll_layout.addWidget(self.insert2_rl)
        scroll_layout.addWidget(self.insert3_tone)
        scroll_layout.addWidget(self.insert4_mono)
        scroll_layout.addWidget(self.insert5_lc)
        scroll_layout.addWidget(self.insert6_color)
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_widget)
        rack_vbox.addWidget(scroll_area)

        rack_outer_hbox.addWidget(self.rack_inner_widget)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.rack_container)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([1100, 380])
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        work_hbox.addWidget(self.splitter)
        root_vbox.addLayout(work_hbox)

        self._update_state_toggle_ui()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #111114;
                color: #dcdce6;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #24242c;
                border: 1px solid #363644;
                border-radius: 5px;
                padding: 2px 10px;
                color: #e0e0ea;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #31313d;
                border-color: #00adb5;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #00adb5;
                color: #111114;
            }
            QPushButton:checked {
                background-color: #1a3c40;
                border-color: #00fff2;
                color: #00fff2;
            }
            QPushButton:disabled {
                color: #555562;
                border-color: #262630;
                background-color: #1a1a20;
            }
            QPushButton#toolIconBtn {
                background-color: #1c1c24;
                border: 1px solid #2c2c38;
                border-radius: 5px;
                color: #c8c8d4;
                font-size: 18px;
                font-weight: normal;
                padding: 0;
            }
            QPushButton#toolIconBtn:hover {
                background-color: #262632;
                border-color: #00adb5;
                color: #ffffff;
            }
            QPushButton#toolIconBtn:checked {
                background-color: #14383c;
                border-color: #00fff2;
                color: #00fff2;
            }
            QPushButton#toolIconBtn:pressed {
                background-color: #00adb5;
                color: #111114;
            }
            QPushButton#toolIconBtn:disabled {
                color: #4a4a58;
                border-color: #22222a;
                background-color: #16161c;
            }
            QPushButton#btnResetImage {
                background-color: transparent;
                border: 1px solid #282f3d;
                border-radius: 5px;
                color: #64748B;
                font-size: 15px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton#btnResetImage:hover {
                background-color: #262B3D;
                border: 1px solid #475569;
                color: #EF4444;
            }
            QPushButton#btnResetImage:pressed {
                background-color: #3b1b20;
                border-color: #EF4444;
                color: #ffffff;
            }
            QPushButton#btnResetImage:disabled {
                background-color: transparent;
                border: 1px solid #1c222d;
                color: #334155;
            }
            QComboBox {
                background-color: #202028;
                border: 1px solid #363644;
                border-radius: 5px;
                padding: 1px 8px;
                color: #00adb5;
                font-size: 11px;
                font-weight: bold;
            }
            QComboBox:disabled {
                background-color: #141419;
                border: 1px solid #1E293B;
                color: #64748B;
            }
            QComboBox::drop-down:disabled {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1c1c24;
                selection-background-color: #00adb5;
                selection-color: #111114;
                border: 1px solid #363644;
            }
            QComboBox QAbstractItemView::separator {
                height: 1px;
                background-color: #363644;
                margin: 4px 6px;
            }
            QSplitter::handle:horizontal {
                background-color: #1c1c26;
                width: 4px;
            }
            QSplitter::handle:horizontal:hover {
                background-color: #00adb5;
            }
            QScrollBar:vertical {
                background: #141418;
                width: 6px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #2b2b36;
                min-height: 20px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00adb5;
            }
            QWidget#DryWetControl {
                background: transparent;
            }
            QCheckBox {
                background: transparent;
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

        # 5. НИЖНИЙ СТАТУС-БАР С ТЕЛЕМЕТРИЕЙ
        self._setup_status_bar()

    def _setup_status_bar(self):
        sb = self.statusBar()
        sb.setFixedHeight(28)
        sb.setSizeGripEnabled(False)
        sb.setStyleSheet("""
            QStatusBar {
                background-color: #0F1117;
                border-top: 1px solid #1E293B;
                color: #94A3B8;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                padding: 0 8px;
            }
            QStatusBar::item {
                border: none;
            }
            QLabel {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                background: transparent;
            }
        """)

        def create_sep():
            sep = QLabel("|")
            sep.setStyleSheet("color: #334155; font-size: 11px; margin: 0 6px;")
            return sep

        # 1. [Статус пайплайна]: "Ready" / "Processing..." / "Render: 18 ms"
        self.lbl_sb_pipeline = QLabel("Ready")
        self.lbl_sb_pipeline.setStyleSheet("color: #00adb5; font-weight: bold;")
        sb.addWidget(self.lbl_sb_pipeline)
        sb.addWidget(create_sep())

        # 2. [Параметры снимка]: "2048×2048 px | 16-bit Float"
        self.lbl_sb_specs = QLabel("No Image Loaded")
        self.lbl_sb_specs.setStyleSheet("color: #cbd5e1; font-weight: 500;")
        sb.addWidget(self.lbl_sb_specs)
        sb.addWidget(create_sep())

        # 3. [Масштаб]: "Zoom: 100%"
        self.lbl_sb_zoom = QLabel("Zoom: 100%")
        self.lbl_sb_zoom.setStyleSheet("color: #94A3B8;")
        sb.addWidget(self.lbl_sb_zoom)
        sb.addWidget(create_sep())

        # 4. [Инспектор под курсором (Live Pixel Probe)]
        self.lbl_sb_coords = QLabel("X: ---, Y: ---")
        self.lbl_sb_coords.setStyleSheet("color: #94A3B8;")
        sb.addWidget(self.lbl_sb_coords)

        self.lbl_sb_probe = QLabel("")
        self.lbl_sb_probe.setStyleSheet("color: #00fff2; font-weight: 600; margin-left: 2px;")
        sb.addWidget(self.lbl_sb_probe)

        # Правый постоянный бренд-лейбл
        lbl_sb_brand = QLabel("AstroRack DAW")
        lbl_sb_brand.setStyleSheet("color: #475569; font-size: 10px; font-weight: 600; margin-right: 4px;")
        sb.addPermanentWidget(lbl_sb_brand)

        # Проксируем основной статус-лейбл для поддержки существующих вызовов
        self.lbl_status = StatusProxy(self.canvas, self.lbl_sb_pipeline)

        # Подключаем сигналы холста
        self.canvas.pixelProbeChanged.connect(self._on_pixel_probe_changed)
        self.canvas.zoomChanged.connect(self._on_canvas_zoom_changed)

    def _on_pixel_probe_changed(self, coords_str: str, val_str: str):
        self.lbl_sb_coords.setText(coords_str)
        if val_str:
            self.lbl_sb_probe.setText(f"| {val_str}")
        else:
            self.lbl_sb_probe.setText("")

    def _on_canvas_zoom_changed(self, zoom_str: str):
        self.lbl_sb_zoom.setText(zoom_str)

    def _update_status_image_specs(self):
        if self._raw_image is not None and self._raw_image.size > 0:
            h, w = self._raw_image.shape[:2]
            if self._raw_image.dtype == np.uint8:
                bit_str = "8-bit Int"
            elif self._raw_image.dtype == np.uint16:
                bit_str = "16-bit Int"
            else:
                bit_str = "16-bit Float"
            self.lbl_sb_specs.setText(f"{w}×{h} px | {bit_str}")
        else:
            self.lbl_sb_specs.setText("No Image Loaded")

    @property
    def sound_muted(self) -> bool:
        return sound_engine.sound_muted

    @sound_muted.setter
    def sound_muted(self, val: bool):
        sound_engine.sound_muted = val
        self._update_sound_button_ui(not val)

    def _show_about_dialog(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def _toggle_sound_effects(self):
        new_state = sound_engine.toggle_enabled()
        self._update_sound_button_ui(new_state)

    def _update_sound_button_ui(self, enabled: bool):
        if not hasattr(self, 'btn_sound') or self.btn_sound is None:
            return
        if enabled:
            self.btn_sound.setText("🔊")
            self.btn_sound.setToolTip("Звуковые эффекты интерфейса: ВКЛ (клик — отключить)")
            self.btn_sound.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: 1px solid transparent;
                    border-radius: 3px;
                    color: #94a3b8;
                    font-size: 11px;
                    padding: 0;
                }
                QPushButton:hover {
                    background-color: #1e2433;
                    border-color: #334155;
                    color: #00fff2;
                }
            """)
        else:
            self.btn_sound.setText("🔇")
            self.btn_sound.setToolTip("Звуковые эффекты интерфейса: ВЫКЛ (клик — включить)")
            self.btn_sound.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: 1px solid transparent;
                    border-radius: 3px;
                    color: #ef4444;
                    font-size: 11px;
                    padding: 0;
                }
                QPushButton:hover {
                    background-color: #1e2433;
                    border-color: #334155;
                    color: #f87171;
                }
            """)

    def _setup_shortcuts(self):
        """Регистрация горячих клавиш для инструментов и управления приложением."""
        QShortcut(QKeySequence("Ctrl+O"), self, self._open_file_dialog)
        QShortcut(QKeySequence("Ctrl+S"), self, self._export_image)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_undo_clicked)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._on_redo_clicked)
        QShortcut(QKeySequence("Z"), self, lambda: self.toolbox.toggle_tool("loupe"))
        QShortcut(QKeySequence("C"), self, lambda: self.toolbox.toggle_tool("crop"))
        QShortcut(QKeySequence("P"), self, lambda: self.toolbox.toggle_tool("psf"))
        QShortcut(QKeySequence("M"), self, lambda: self.toolbox.toggle_tool("ruler"))
        QShortcut(QKeySequence("L"), self, lambda: self.toolbox.toggle_tool("profile"))
        QShortcut(QKeySequence("R"), self, lambda: self._apply_geometry(AstroEngine.rotate_90_cw, "Rotate 90° CW"))
        QShortcut(QKeySequence("Shift+R"), self, lambda: self._apply_geometry(AstroEngine.rotate_90_ccw, "Rotate 90° CCW"))
        QShortcut(QKeySequence("H"), self, lambda: self._apply_geometry(AstroEngine.flip_horizontal, "Flip Horizontal"))
        QShortcut(QKeySequence("V"), self, lambda: self._apply_geometry(AstroEngine.flip_vertical, "Flip Vertical"))
        QShortcut(QKeySequence("Space"), self, lambda: self.btn_global_bypass.toggle())
        QShortcut(QKeySequence("S"), self, lambda: self.btn_split_view.toggle() if self.btn_split_view.isEnabled() else None)
        QShortcut(QKeySequence("F1"), self, self._show_hotkeys_dialog)
        QShortcut(QKeySequence("Escape"), self, self._reset_active_tools)

    def _reset_active_tools(self):
        """Полный сброс активного инструмента (ESC): отмена crop, лупы, профиля, psf, ruler и закрытие инспекторов."""
        self.toolbox.clear_tool()
        self.canvas.reset_to_default_tool()
        self.canvas.clear_ruler()
        self.canvas.clear_psf_marker()
        if hasattr(self, '_profile_dialog') and self._profile_dialog is not None:
            self._profile_dialog.hide()
        if hasattr(self, '_psf_dialog') and self._psf_dialog is not None:
            self._psf_dialog.hide()
        base_status = f"{self._current_filename} ({self._raw_image.shape[1]}x{self._raw_image.shape[0]})" if self._raw_image is not None else "No image loaded"
        self.lbl_status.setText(base_status)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._reset_active_tools()
            event.accept()
            return
        super().keyPressEvent(event)

    def _show_hotkeys_dialog(self):
        """Окно справки по горячим клавишам приложения в тёмном стиле DAW."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Справка по горячим клавишам")
        msg.setIcon(QMessageBox.Icon.NoIcon)
        msg.setTextFormat(Qt.TextFormat.RichText)

        table_html = """
        <div style='font-size: 11px; line-height: 150%; color: #dcdce6; padding: 6px 14px;'>
            <h3 style='color: #00fff2; margin-top: 0; margin-bottom: 8px;'>Горячие клавиши AstroRack DAW</h3>
            <table cellpadding='4' cellspacing='0' style='width: 100%; border-collapse: collapse;'>
                <tr style='border-bottom: 1px solid #363648;'>
                    <th align='left' style='color: #e5a93b;'>Клавиша</th>
                    <th align='left' style='color: #e5a93b;'>Действие</th>
                </tr>
                <tr><td><b>Z</b></td><td>Инспектор лупы (Loupe Tool / Zoom)</td></tr>
                <tr><td><b>Alt (зажать)</b></td><td>Временная быстрая лупа под курсором</td></tr>
                <tr><td><b>C</b></td><td>Инструмент кадрирования (Crop Tool)</td></tr>
                <tr><td><b>P</b></td><td>Инспектор PSF / FWHM (измерение профиля звезды и кратеров)</td></tr>
                <tr><td><b>M</b></td><td>Интерактивная линейка расстояний и углов (Pixel & Angular Ruler)</td></tr>
                <tr><td><b>L</b></td><td>Инспектор среза яркости (Profile Line Scan)</td></tr>
                <tr><td><b>Enter</b></td><td>Применить кадрирование</td></tr>
                <tr><td><b>Esc</b></td><td>Отменить кадрирование / Сбросить активный инструмент</td></tr>
                <tr><td><b>R</b></td><td>Поворот кадра на 90° по часовой стрелке</td></tr>
                <tr><td><b>Shift + R</b></td><td>Поворот кадра на 90° против часовой стрелки</td></tr>
                <tr><td><b>H</b></td><td>Отразить кадр по горизонтали (Flip H)</td></tr>
                <tr><td><b>V</b></td><td>Отразить кадр по вертикали (Flip V)</td></tr>
                <tr><td><b>Space (Пробел)</b></td><td>Master Bypass (мгновенное сравнение До/После)</td></tr>
                <tr><td><b>Ctrl + O</b></td><td>Открыть изображение (Open Image)</td></tr>
                <tr><td><b>Ctrl + S</b></td><td>Экспортировать результат (Export Image)</td></tr>
                <tr><td><b>Ctrl + Z</b></td><td>Отменить действие (Undo)</td></tr>
                <tr><td><b>Ctrl + Y</b></td><td>Повторить действие (Redo)</td></tr>
                <tr><td><b>F1</b></td><td>Открыть справку по горячим клавишам</td></tr>
            </table>
        </div>
        """
        msg.setText(table_html)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        btn_ok = msg.button(QMessageBox.StandardButton.Ok)
        if btn_ok:
            btn_ok.setText("Закрыть")
        msg.setFixedWidth(460)
        for lbl in msg.findChildren(QLabel):
            lbl.setWordWrap(True)
            lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #16161e;
                border: 1px solid #2e2e3e;
                border-radius: 8px;
            }
            QLabel {
                background: transparent;
                border: none;
                min-width: 420px;
                max-width: 420px;
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
            }
        """)
        msg.exec()

    def _get_panel_width(self) -> int:
        sizes = self.splitter.sizes()
        return sizes[1] if len(sizes) > 1 else 380

    def _set_panel_width(self, w: int):
        sizes = self.splitter.sizes()
        total_w = sum(sizes)
        if total_w <= 0:
            total_w = max(500, self.width() - 50)
        w = max(16, min(total_w - 100, w))
        self.splitter.setSizes([total_w - w, w])

    panelWidth = pyqtProperty(int, _get_panel_width, _set_panel_width)

    def _on_splitter_moved(self, pos: int, index: int):
        if not self._panel_collapsed:
            sizes = self.splitter.sizes()
            if len(sizes) > 1 and sizes[1] >= 200:
                self._last_panel_width = sizes[1]

    def _toggle_right_panel(self):
        if self._panel_anim.state() == QPropertyAnimation.State.Running:
            return

        current_w = self._get_panel_width()
        if not self._panel_collapsed:
            self._target_collapsed = True
            self._last_panel_width = max(340, current_w)
            self.rack_container.setMinimumWidth(0)
            self.rack_container.setMaximumWidth(10000)
            self._panel_anim.stop()
            self._panel_anim.setStartValue(current_w)
            self._panel_anim.setEndValue(16)
            self._panel_anim.start()
        else:
            self._target_collapsed = False
            self.rack_inner_widget.setVisible(True)
            self.rack_container.setMinimumWidth(0)
            self.rack_container.setMaximumWidth(10000)
            self._panel_anim.stop()
            self._panel_anim.setStartValue(current_w)
            self._panel_anim.setEndValue(self._last_panel_width)
            self._panel_anim.start()

    def _on_panel_anim_finished(self):
        if getattr(self, '_target_collapsed', False):
            self._panel_collapsed = True
            self.rack_inner_widget.setVisible(False)
            self.rack_container.setFixedWidth(16)
            self._set_panel_width(16)
            self.btn_panel_toggle.setText("◄")
            self.btn_panel_toggle.setToolTip("Развернуть панель настроек")
            if self.splitter.handle(1):
                self.splitter.handle(1).setEnabled(False)
        else:
            self._panel_collapsed = False
            self.rack_container.setMinimumWidth(340)
            self.rack_container.setMaximumWidth(700)
            self.rack_inner_widget.setVisible(True)
            self._set_panel_width(self._last_panel_width)
            self.btn_panel_toggle.setText("►")
            self.btn_panel_toggle.setToolTip("Свернуть панель настроек")
            if self.splitter.handle(1):
                self.splitter.handle(1).setEnabled(True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, '_panel_collapsed', False):
            self._set_panel_width(16)

    def _toggle_ab_state(self):
        target = 'B' if self._active_ab == 'A' else 'A'
        self._switch_ab_state(target)

    def _update_state_toggle_ui(self):
        if not hasattr(self, 'btn_state_toggle'):
            return
        theme = THEMES.get(self._active_ab, THEMES['A'])
        if self._active_ab == 'A':
            self.btn_state_toggle.setText("State: [ A ]")
            self.btn_state_toggle.setToolTip("Текущий снимок: State A (Cyan Accent). Клик — переключить на State B")
            self.btn_state_toggle.setStyleSheet(f"""
                QPushButton {{
                    background-color: {theme['bg_accent']};
                    border: 1px solid {theme['bright']};
                    color: {theme['bright']};
                    font-weight: bold;
                    font-size: 11px;
                    border-radius: 4px;
                    padding: 2px 10px;
                }}
                QPushButton:hover {{
                    background-color: {theme['bg_hover']};
                    color: #ffffff;
                }}
            """)
        else:
            self.btn_state_toggle.setText("State: [ B ]")
            self.btn_state_toggle.setToolTip("Текущий снимок: State B (Amber Accent). Клик — переключить на State A")
            self.btn_state_toggle.setStyleSheet(f"""
                QPushButton {{
                    background-color: {theme['bg_accent']};
                    border: 1px solid {theme['bright']};
                    color: {theme['bright']};
                    font-weight: bold;
                    font-size: 11px;
                    border-radius: 4px;
                    padding: 2px 10px;
                }}
                QPushButton:hover {{
                    background-color: {theme['bg_hover']};
                    color: #ffffff;
                }}
            """)

        self._apply_state_theme(self._active_ab)

    def _apply_state_theme(self, state_name: str):
        theme = THEMES.get(state_name, THEMES['A'])
        bright = theme.get('bright', '#00fff2')
        primary = theme.get('primary', '#00adb5')
        bg_accent = theme.get('bg_accent', '#14383c')

        # 1. Инсерты рэка
        for ins in [
            getattr(self, 'insert1_wavelet', None),
            getattr(self, 'insert2_rl', None),
            getattr(self, 'insert3_tone', None),
            getattr(self, 'insert4_mono', None),
            getattr(self, 'insert5_lc', None),
            getattr(self, 'insert6_color', None)
        ]:
            if ins is not None and hasattr(ins, 'apply_theme'):
                ins.apply_theme(theme)

        # 2. Холст (AstroCanvas, Welcome Canvas, Split View)
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'apply_theme'):
            self.canvas.apply_theme(theme)

        # 3. Левая панель инструментов (ToolboxPanel)
        if hasattr(self, 'toolbox') and hasattr(self.toolbox, 'apply_theme'):
            self.toolbox.apply_theme(theme)

        # 3.1. Анализатор спектра и шапка гистограммы
        if hasattr(self, 'hist_header') and self.hist_header is not None:
            self.hist_header.apply_theme(theme)
        if hasattr(self, 'histogram') and hasattr(self.histogram, 'apply_theme'):
            self.histogram.apply_theme(theme)

        # 4. Кнопки верхнего тулбара (hover border, checked styles)
        tool_btns = [
            getattr(self, 'btn_open', None),
            getattr(self, 'btn_export', None),
            getattr(self, 'btn_split_view', None),
            getattr(self, 'btn_invert_split', None),
            getattr(self, 'btn_split_orient', None),
            getattr(self, 'btn_undo', None),
            getattr(self, 'btn_redo', None),
        ]
        btn_style = f"""
            QPushButton {{
                background-color: #24242c;
                border: 1px solid #363644;
                border-radius: 5px;
                padding: 2px 10px;
                color: #e0e0ea;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #31313d;
                border-color: {bright};
                color: #ffffff;
            }}
            QPushButton:pressed {{
                background-color: {primary};
                color: #111114;
            }}
            QPushButton:checked {{
                background-color: {bg_accent};
                border-color: {bright};
                color: {bright};
            }}
            QPushButton:disabled {{
                color: #555562;
                border-color: #262630;
                background-color: #1a1a20;
            }}
            QPushButton#toolIconBtn {{
                background-color: #1c1c24;
                border: 1px solid #2c2c38;
                border-radius: 5px;
                color: #c8c8d4;
                font-size: 18px;
                font-weight: normal;
                padding: 0;
            }}
            QPushButton#toolIconBtn:hover {{
                background-color: #262632;
                border-color: {bright};
                color: #ffffff;
            }}
            QPushButton#toolIconBtn:checked {{
                background-color: {bg_accent};
                border-color: {bright};
                color: {bright};
            }}
            QPushButton#toolIconBtn:pressed {{
                background-color: {primary};
                color: #111114;
            }}
            QPushButton#toolIconBtn:disabled {{
                color: #4a4a58;
                border-color: #22222a;
                background-color: #16161c;
            }}
        """
        for btn in tool_btns:
            if btn is not None:
                btn.setStyleSheet(btn_style)

        if hasattr(self, 'btn_about') and self.btn_about is not None:
            self.btn_about.setStyleSheet(f"""
                QPushButton#toolIconBtn {{
                    background-color: #1c1c24;
                    border: 1px solid #2c2c38;
                    border-radius: 5px;
                    color: {primary};
                    font-size: 15px;
                    font-weight: bold;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    padding: 0;
                    text-align: center;
                }}
                QPushButton#toolIconBtn:hover {{
                    background-color: #262632;
                    border-color: {bright};
                    color: #ffffff;
                }}
                QPushButton#toolIconBtn:pressed {{
                    background-color: {primary};
                    color: #111114;
                }}
            """)

        if hasattr(self, 'btn_hotkeys') and self.btn_hotkeys is not None:
            self.btn_hotkeys.setStyleSheet(f"""
                QPushButton#toolIconBtn {{
                    background-color: #1c1c24;
                    border: 1px solid #2c2c38;
                    border-radius: 5px;
                    color: #c8c8d4;
                    font-size: 16px;
                    font-family: 'Segoe UI Symbol', 'Segoe UI', Arial, sans-serif;
                    padding: 0;
                    text-align: center;
                }}
                QPushButton#toolIconBtn:hover {{
                    background-color: #262632;
                    border-color: {bright};
                    color: #ffffff;
                }}
                QPushButton#toolIconBtn:pressed {{
                    background-color: {primary};
                    color: #111114;
                }}
            """)

        if hasattr(self, 'combo_presets') and self.combo_presets is not None:
            self.combo_presets.setStyleSheet(f"""
                QComboBox {{
                    background-color: #202028;
                    border: 1px solid #363644;
                    border-radius: 5px;
                    padding: 1px 8px;
                    color: {primary};
                    font-size: 11px;
                    font-weight: bold;
                }}
                QComboBox:disabled {{
                    background-color: #141419;
                    border: 1px solid #1E293B;
                    color: #64748B;
                }}
                QComboBox::drop-down:disabled {{
                    border: none;
                }}
                QComboBox QAbstractItemView {{
                    background-color: #1c1c24;
                    selection-background-color: {primary};
                    selection-color: #111114;
                    border: 1px solid #363644;
                }}
            """)

        if hasattr(self, 'btn_panel_toggle') and self.btn_panel_toggle is not None:
            self.btn_panel_toggle.setStyleSheet(f"""
                QPushButton {{
                    background-color: #1e1e28;
                    border: 1px solid #363648;
                    border-radius: 5px;
                    color: {primary};
                    font-size: 10px;
                    font-weight: bold;
                    padding: 0;
                }}
                QPushButton:hover {{
                    background-color: {primary};
                    border-color: {bright};
                    color: #111114;
                }}
            """)

        if hasattr(self, 'btn_reset_image') and self.btn_reset_image is not None:
            self.btn_reset_image.setStyleSheet("""
                QPushButton#btnResetImage {
                    background-color: transparent;
                    border: 1px solid #282f3d;
                    border-radius: 5px;
                    color: #64748B;
                    font-size: 15px;
                    font-weight: bold;
                    padding: 0;
                }
                QPushButton#btnResetImage:hover {
                    background-color: #262B3D;
                    border: 1px solid #475569;
                    color: #EF4444;
                }
                QPushButton#btnResetImage:pressed {
                    background-color: #3b1b20;
                    border-color: #EF4444;
                    color: #ffffff;
                }
                QPushButton#btnResetImage:disabled {
                    background-color: transparent;
                    border: 1px solid #1c222d;
                    color: #334155;
                }
            """)

    def _on_reset_image_clicked(self, checked: bool = False, skip_confirm: bool = False):
        """Защита от случайного сброса через кастомный модальный диалог подтверждения."""
        if self._raw_image is None or getattr(self._raw_image, 'size', 0) == 0:
            return

        if skip_confirm:
            self._reset_to_welcome_state()
            return

        dlg = ResetConfirmDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._reset_to_welcome_state()

    def _reset_to_welcome_state(self):
        """Полный сброс текущего изображения, отмена пайплайна и возврат к Welcome State."""
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.cancel_current()
        if hasattr(self, '_debounce_timer'):
            self._debounce_timer.stop()
        if hasattr(self, '_final_render_timer'):
            self._final_render_timer.stop()
        if hasattr(self, '_hist_timer'):
            self._hist_timer.stop()

        if hasattr(self, 'btn_split_view') and self.btn_split_view.isChecked():
            self.btn_split_view.setChecked(False)

        self._raw_image = None
        self._current_processed_image = None
        self._current_filename = ""
        self._last_dispatched_params = None

        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.show_welcome_state()

        self._reset_all_to_flat()

        if hasattr(self, 'history') and self.history is not None:
            self.history.clear()
        self._update_undo_redo_ui()

        if hasattr(self, 'histogram') and self.histogram is not None:
            self.histogram.clear()

        self._update_status_image_specs()
        if hasattr(self, 'lbl_sb_pipeline'):
            self.lbl_sb_pipeline.setText("Ready")

        self._update_canvas_guard_state()

    def _update_canvas_guard_state(self):
        """Блокировка органов управления при отсутствии изображения (Empty Canvas Guard)."""
        has_image = self._raw_image is not None and getattr(self._raw_image, 'size', 0) > 0

        # 1. Левый вертикальный тулбар инструментов
        if hasattr(self, 'toolbox') and self.toolbox is not None:
            self.toolbox.setEnabled(has_image)

        # 2. Кнопки верхней панели
        for btn in [
            getattr(self, 'btn_export', None),
            getattr(self, 'btn_reset_image', None),
            getattr(self, 'btn_split_view', None),
            getattr(self, 'btn_invert_split', None),
            getattr(self, 'btn_split_orient', None),
            getattr(self, 'combo_presets', None),
        ]:
            if btn is not None:
                btn.setEnabled(has_image)

        # Undo / Redo
        if hasattr(self, 'history') and self.history is not None:
            can_u = self.history.can_undo() and has_image
            can_r = self.history.can_redo() and has_image
            if hasattr(self, 'btn_undo') and self.btn_undo is not None:
                self.btn_undo.setEnabled(can_u)
            if hasattr(self, 'btn_redo') and self.btn_redo is not None:
                self.btn_redo.setEnabled(can_r)

        # 3. Рэк инсертов, гистограмма, переключатели стейтов и Master Bypass
        for widget in [
            getattr(self, 'insert1_wavelet', None),
            getattr(self, 'insert2_rl', None),
            getattr(self, 'insert3_tone', None),
            getattr(self, 'insert4_mono', None),
            getattr(self, 'insert5_lc', None),
            getattr(self, 'insert6_color', None),
            getattr(self, 'histogram', None),
            getattr(self, 'btn_state_toggle', None),
            getattr(self, 'btn_copy_ab', None),
            getattr(self, 'btn_copy_ba', None),
            getattr(self, 'btn_global_bypass', None),
        ]:
            if widget is not None:
                widget.setEnabled(has_image)

        if hasattr(self, 'hist_header') and self.hist_header is not None:
            self.hist_header.set_canvas_enabled(has_image)

        # 4. Всегда активные элементы
        if hasattr(self, 'btn_open') and self.btn_open is not None:
            self.btn_open.setEnabled(True)
        if hasattr(self, 'btn_hotkeys') and self.btn_hotkeys is not None:
            self.btn_hotkeys.setEnabled(True)
        if hasattr(self, 'btn_about') and self.btn_about is not None:
            self.btn_about.setEnabled(True)
        if hasattr(self, 'btn_sound') and self.btn_sound is not None:
            self.btn_sound.setEnabled(True)

    def _switch_ab_state(self, target: str):
        if self._active_ab == target:
            self._update_state_toggle_ui()
            return

        current_dict = self._get_current_state_dict()
        if self._active_ab == 'A':
            self._state_a = current_dict
            self._active_ab = 'B'
            if self._state_b is not None:
                self._apply_preset_dict(self._state_b)
        else:
            self._state_b = current_dict
            self._active_ab = 'A'
            if self._state_a is not None:
                self._apply_preset_dict(self._state_a)

        self._update_state_toggle_ui()
        self._sync_histogram_visors()
        self.lbl_status.setText(f"Active: State {self._active_ab}")

    def _copy_a_to_b(self):
        cur = self._get_current_state_dict()
        if self._active_ab == 'A':
            self._state_a = cur
            self._state_b = cur.copy()
        else:
            if self._state_a is not None:
                self._state_b = self._state_a.copy()
        self.lbl_status.setText("State A скопирован в State B (A→B)")

    def _copy_b_to_a(self):
        cur = self._get_current_state_dict()
        if self._active_ab == 'B':
            self._state_b = cur
            self._state_a = cur.copy()
        else:
            if self._state_b is not None:
                self._state_a = self._state_b.copy()
        self.lbl_status.setText("State B скопирован в State A (B→A)")

    def _on_toolbox_tool_changed(self, tool: str):
        self.canvas.set_loupe_mode(tool == "loupe")
        self.canvas.set_crop_mode(tool == "crop")
        self.canvas.set_profile_mode(tool == "profile")
        self.canvas.set_psf_mode(tool == "psf")
        self.canvas.set_ruler_mode(tool == "ruler")

        if tool == "profile":
            if self._profile_dialog is None:
                self._profile_dialog = ProfileMonitor(self)
            self._profile_dialog.show()
            self.lbl_status.setText("Проведите линию среза на кадре мышью (L)...")
        else:
            if self._profile_dialog is not None:
                self._profile_dialog.hide()

        if tool != "psf" and self._psf_dialog is not None:
            self._psf_dialog.hide()

        if tool == "loupe":
            self.lbl_status.setText("Лупа: наведите курсор на кадр (Z / Alt)")
        elif tool == "crop":
            self.lbl_status.setText("Кадрирование: потяните за рамку (C) — Enter: подтвердить, Esc: отмена")
        elif tool == "psf":
            self.lbl_status.setText("Инспектор PSF: кликните на звезду или кратер для измерения FWHM (P)")
        elif tool == "ruler":
            self.lbl_status.setText("Линейка: зажмите ЛКМ и протяните линию (M). ПКМ по кнопке — оптика")
        elif tool == "":
            self.lbl_status.setText(self._current_filename or "No image loaded")

    def _on_psf_point_clicked(self, pos: QPointF):
        img = self._current_processed_image if self._current_processed_image is not None else self._raw_image
        if img is None:
            return
        res = AstroEngine.compute_psf_fwhm(img, pos.x(), pos.y())
        if res is None:
            self.lbl_status.setText("PSF: Точка вне кадра или сигнал слишком слабый")
            return

        if self._psf_dialog is None:
            self._psf_dialog = PsfInspectorDialog(self)
            self._psf_dialog.applyPsfRequested.connect(self._apply_psf_to_deconv)
            self._psf_dialog.finished.connect(self._on_psf_dialog_closed)

        self._psf_dialog.update_metrics(res)
        self._psf_dialog.show()
        self._psf_dialog.raise_()
        self.lbl_status.setText(f"PSF: FWHM={res['fwhm']} px, σ={res['sigma']} px, Peak={res['peak']} | ESC для сброса")

    def _on_psf_dialog_closed(self, _result: int = 0):
        """Закрытие инспектора PSF (×, Esc или Apply): инструмент PSF снимается, активным становится Pan."""
        if self.toolbox.btn_psf.isChecked() or self.canvas._psf_mode:
            self._reset_active_tools()
        msg = getattr(self, '_psf_apply_msg', None)
        if msg:
            self.lbl_status.setText(msg)
            self._psf_apply_msg = None

    def _apply_psf_to_deconv(self, sigma: float):
        """Применение измеренной сигмы PSF в радиус деконволюции Ричардсона-Люси (Инсерт 2)."""
        self.insert2_rl.row_sigma.set_value(sigma)
        if not self.insert2_rl.is_enabled():
            self.insert2_rl.set_enabled(True)
        self.trigger_render_immediate()
        self._psf_apply_msg = f"PSF σ={sigma:.2f} px подставлена в Deconvolution (Insert 2)"
        self.lbl_status.setText(self._psf_apply_msg)
        self._record_current_state(f"Apply PSF σ={sigma:.2f} px")

    def _show_optics_dialog(self):
        """Вызов диалога настройки оптики телескопа для линейки."""
        dlg = TelescopeOpticsDialog(self.canvas._telescope_focal_len, self.canvas._telescope_pixel_size, self)
        if dlg.exec():
            f_mm, p_um = dlg.get_values()
            self.canvas.set_telescope_params(f_mm, p_um)
            scale_val = 206.265 * p_um / f_mm
            self.lbl_status.setText(f"Оптика: F={f_mm:.0f} мм, Pixel={p_um:.2f} µm ({scale_val:.2f} arcsec/px)")

    def _on_ruler_measured(self, dist: float, angle: float, summary: str):
        self.lbl_status.setText(f"Ruler: {summary} | Esc для сброса")

    def _on_crop_cancelled(self):
        self._reset_active_tools()

    def _on_profile_line_selected(self, p1: QPointF, p2: QPointF):
        if self._current_processed_image is None or self._profile_dialog is None:
            return

        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        length = int(np.hypot(x2 - x1, y2 - y1))

        if length < 2:
            return

        img = self._current_processed_image
        if img.ndim == 3:
            lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            lum = img

        xs = np.linspace(x1, x2, length)
        ys = np.linspace(y1, y2, length)
        coords = np.vstack((ys, xs))
        profile_data = map_coordinates(lum, coords, order=1, mode='nearest')
        self._profile_dialog.update_profile(profile_data)

    def _on_rgb_shifts_computed(self, dx_r, dy_r, dx_b, dy_b):
        self.insert4_mono.set_shifts_label(dx_r, dy_r, dx_b, dy_b)

    def _get_current_state_dict(self) -> dict:
        state = {
            'global_bypass': self.btn_global_bypass.isChecked(),
            'insert1_enabled': self.insert1_wavelet.is_enabled(),
            'insert2_enabled': self.insert2_rl.is_enabled(),
            'insert3_enabled': self.insert3_tone.is_enabled(),
            'insert4_enabled': self.insert4_mono.is_enabled(),
            'insert5_enabled': self.insert5_lc.is_enabled(),
            'insert6_enabled': self.insert6_color.is_enabled(),
        }
        state.update(self.insert1_wavelet.get_values())
        state.update(self.insert2_rl.get_values())
        state.update(self.insert3_tone.get_values())
        state.update(self.insert4_mono.get_values())
        state.update(self.insert5_lc.get_values())
        state.update(self.insert6_color.get_values())
        return state

    def _record_current_state(self, action_name: str = "Parameters"):
        state = self._get_current_state_dict()
        state['_action_name'] = action_name
        if self._raw_image is not None:
            if action_name in ("Crop", "Rotate 90° CW", "Rotate 90° CCW", "Flip Horizontal", "Flip Vertical", "Open Image", "Initial State"):
                state['raw_image'] = self._raw_image.copy()
            else:
                state['raw_image'] = self._raw_image
        else:
            state['raw_image'] = None

        state['canvas_transform'] = self.canvas.transform()
        self.history.push(state, action_name)
        self._update_undo_redo_ui()

    def _update_undo_redo_ui(self):
        can_u = self.history.can_undo()
        can_r = self.history.can_redo()
        self.btn_undo.setEnabled(can_u)
        self.btn_redo.setEnabled(can_r)

        if can_u:
            u_act = self.history.undo_action_name()
            self.btn_undo.setToolTip(f"Undo: {u_act} (Ctrl+Z)")
        else:
            self.btn_undo.setToolTip("Undo (Ctrl+Z)")

        if can_r:
            r_act = self.history.redo_action_name()
            self.btn_redo.setToolTip(f"Redo: {r_act} (Ctrl+Y)")
        else:
            self.btn_redo.setToolTip("Redo (Ctrl+Y)")

    def _on_undo_clicked(self):
        state = self.history.undo()
        if state is not None:
            self._apply_preset_dict(state)
            self._update_undo_redo_ui()

    def _on_redo_clicked(self):
        state = self.history.redo()
        if state is not None:
            self._apply_preset_dict(state)
            self._update_undo_redo_ui()

    def _on_crop_applied(self, r: QRectF):
        if self._raw_image is None:
            return

        x1 = int(round(max(0, r.left())))
        y1 = int(round(max(0, r.top())))
        x2 = int(round(min(self._raw_image.shape[1], r.right())))
        y2 = int(round(min(self._raw_image.shape[0], r.bottom())))

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            QMessageBox.warning(self, "Crop", "Выделенная область слишком мала.")
            return

        if self._raw_image.ndim == 2:
            self._raw_image = self._raw_image[y1:y2, x1:x2].copy()
        else:
            self._raw_image = self._raw_image[y1:y2, x1:x2, :].copy()

        self._reset_active_tools()
        self._last_dispatched_params = None
        self.canvas.set_raw_image(self._raw_image)
        self.canvas.fit_current_image()
        self.worker.set_source_image(self._raw_image)
        self.lbl_status.setText(f"Кадрировано: [{self._raw_image.shape[1]}x{self._raw_image.shape[0]}]")
        self._update_status_image_specs()
        self.canvas.setFocus()
        self.trigger_render_immediate()
        self._record_current_state("Crop")

    def _apply_geometry(self, transform_fn, label: str):
        if self._raw_image is None:
            return

        self._raw_image = transform_fn(self._raw_image)
        self._last_dispatched_params = None
        self.canvas.set_raw_image(self._raw_image)
        self.worker.set_source_image(self._raw_image)

        if self._current_processed_image is not None:
            self._current_processed_image = transform_fn(self._current_processed_image)
            self.canvas.set_processed_image(self._current_processed_image)
            self._flush_histogram_update()

        self.canvas.fit_current_image()
        self.lbl_status.setText(
            f"{label}: [{self._raw_image.shape[1]}x{self._raw_image.shape[0]}]"
        )
        self._update_status_image_specs()
        self.trigger_render_immediate()
        self._record_current_state(label)

    def _reset_all_to_flat(self):
        inserts = [
            self.insert1_wavelet, self.insert2_rl, self.insert3_tone,
            self.insert4_mono, self.insert5_lc, self.insert6_color
        ]
        for insert in inserts:
            insert.blockSignals(True)
            insert.header.chk_bypass.blockSignals(True)
            insert.set_enabled(False)
            insert.header.chk_bypass.blockSignals(False)

        self.insert1_wavelet.set_flat()
        self.insert2_rl.set_flat()

        self.insert3_tone.row_black.set_param_enabled(False)
        self.insert3_tone.row_gamma.set_param_enabled(False)
        self.insert3_tone.row_contrast.set_param_enabled(False)

        self.insert4_mono.btn_rgb_align.blockSignals(True)
        self.insert4_mono.chk_green.blockSignals(True)
        self.insert4_mono.btn_rgb_align.setChecked(False)
        self.insert4_mono.chk_green.setChecked(False)
        self.insert4_mono.btn_rgb_align.blockSignals(False)
        self.insert4_mono.chk_green.blockSignals(False)

        self.insert5_lc.set_flat()
        self.insert6_color.set_flat()

        for insert in inserts:
            insert.blockSignals(False)

        self._sync_histogram_visors()

        flat_idx = self.combo_presets.findText("Flat State (All Bypassed)")
        if flat_idx >= 0:
            self._last_preset_index = flat_idx
            self.combo_presets.blockSignals(True)
            self.combo_presets.setCurrentIndex(flat_idx)
            self.combo_presets.blockSignals(False)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                self._load_image_file(file_path)
                break

    def _open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Lunar/Planetary Image", self._last_open_dir,
            "Astro & Photo Images (*.tif *.tiff *.png *.jpg *.jpeg *.fit *.fits);;All Files (*)"
        )
        if file_path:
            self._last_open_dir = os.path.dirname(file_path)
            self._load_image_file(file_path)

    def open_image(self, file_path: str):
        """Публичный интерфейс загрузки астроснимка в рабочий конвейер."""
        self._load_image_file(file_path)

    def _load_image_file(self, file_path: str):
        try:
            self._do_load_image_file(file_path)
        except Exception as e:
            traceback.print_exc()
            filename = os.path.basename(file_path) if file_path else "Unknown"
            err_msg = f"Ошибка открытия файла '{filename}': {e}"
            self.lbl_status.setText(err_msg)
            QMessageBox.critical(self, "Load Error", f"{err_msg}\n\nПодробности см. в консоли терминала.")

    def _do_load_image_file(self, file_path: str):
        if not file_path or not os.path.isfile(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        # 1. Отмена текущих фоновых расчетов и остановка таймеров
        self.worker.cancel_current()
        self._debounce_timer.stop()
        self._final_render_timer.stop()
        self._hist_timer.stop()

        # 2. Сброс Split View в стандартное состояние (Single View)
        if self.btn_split_view.isChecked():
            self.btn_split_view.blockSignals(True)
            self.btn_split_view.setChecked(False)
            self.btn_split_view.blockSignals(False)
        self.canvas.set_split_view_active(False)
        self.btn_invert_split.setVisible(False)
        self.btn_split_orient.setVisible(False)

        # 3. Сброс активных оверлеев и инструментов на холсте
        self._reset_active_tools()

        # 4. Загрузка файла с диска
        img = load_image(file_path)
        if img is None or img.size == 0:
            raise ValueError(f"Не удалось декодировать изображение из файла: {file_path}")

        # 5. Блокировка сигналов от слайдеров и инсертов (QSignalBlocker)
        blockers = [
            QSignalBlocker(self.insert1_wavelet),
            QSignalBlocker(self.insert2_rl),
            QSignalBlocker(self.insert3_tone),
            QSignalBlocker(self.insert4_mono),
            QSignalBlocker(self.insert5_lc),
            QSignalBlocker(self.insert6_color),
            QSignalBlocker(self.btn_global_bypass),
            QSignalBlocker(self.combo_presets),
            QSignalBlocker(self.btn_state_a),
            QSignalBlocker(self.btn_state_b),
        ]
        for ins in [
            self.insert1_wavelet, self.insert2_rl, self.insert3_tone,
            self.insert4_mono, self.insert5_lc, self.insert6_color
        ]:
            blockers.append(QSignalBlocker(ins.header.chk_bypass))
            if hasattr(ins.header, 'dry_wet_ctrl') and ins.header.dry_wet_ctrl is not None:
                blockers.append(QSignalBlocker(ins.header.dry_wet_ctrl))

        # 6. Сброс холста и освобождение старых буферов
        self.canvas.reset_for_new_image()

        self._raw_image = img.copy()
        self._current_processed_image = None
        self._last_dispatched_params = None
        self._current_filename = os.path.basename(file_path)

        self.canvas.set_raw_image(self._raw_image)
        self.worker.set_source_image(self._raw_image)
        self.lbl_status.setText(f"{self._current_filename} [{img.shape[1]}x{img.shape[0]}]")
        self._update_status_image_specs()

        self._reset_all_to_flat()
        self.history.clear()
        self._record_current_state("Open Image")
        self._state_a = self._get_current_state_dict()
        self._state_b = self._get_current_state_dict()
        self._update_canvas_guard_state()

        # Освобождаем блокеры сигналов
        del blockers

        # 7. Запуск первичного рендеринга
        self.trigger_render_immediate()

    def _export_image(self):
        if self._current_processed_image is None:
            QMessageBox.information(self, "Export", "Нет обработанного кадра для сохранения.")
            return

        default_name = f"processed_{os.path.splitext(self._current_filename)[0]}.tif"
        default_path = os.path.join(self._last_save_dir, default_name)

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", default_path,
            "16-bit TIFF (*.tif);;16-bit PNG (*.png);;8-bit JPEG (*.jpg)"
        )
        if not save_path:
            return

        self._last_save_dir = os.path.dirname(save_path)

        try:
            success = save_image(save_path, self._current_processed_image)
            if success:
                self.lbl_status.setText(f"Exported: {os.path.basename(save_path)}")
                sound_engine.play_shutter()
            else:
                QMessageBox.critical(self, "Export Error", "Не удалось кодировать изображение.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Ошибка записи файла:\n{str(e)}")

    def _is_dragging_slider(self) -> bool:
        return bool(QApplication.mouseButtons() & Qt.MouseButton.LeftButton)

    def update_insert_param(self, insert_id: int, param_name: str, value: float):
        """Прямое обновление параметра инсерта в пайплайне и запуск рендера."""
        insert_map = {
            1: ('w_wet_mix', self.insert1_wavelet),
            2: ('rl_wet_mix', self.insert2_rl),
            5: ('lc_wet_mix', self.insert5_lc),
            6: ('cs_wet_mix', self.insert6_color),
        }
        if insert_id in insert_map:
            key, ins = insert_map[insert_id]
            if param_name in ('wet_mix', key):
                ratio = float(value)
                if ratio > 1.0:
                    ratio /= 100.0
                if abs(ins.get_dry_wet_ratio() - ratio) > 1e-4:
                    ins.set_dry_wet_ratio(ratio)
        self.schedule_render()

    def schedule_render(self):
        # Троттлинг (не более ~28 кадров/с во время ведения ползунка) + отложенный чистовой рендер
        if not self._debounce_timer.isActive():
            self._debounce_timer.start()
        self._final_render_timer.start()

    def _on_debounce_timeout(self):
        is_dragging = self._is_dragging_slider()
        self._execute_render(preview=is_dragging)

    def _on_parameter_committed(self):
        self._debounce_timer.stop()
        self._final_render_timer.stop()
        self._execute_render(preview=False)
        self._record_current_state("Parameters")

    def _on_wavelet_layers_changed(self, action_name: str):
        """Добавление/удаление слоя вейвлета: немедленный рендер и отдельная запись в Undo/Redo."""
        self._debounce_timer.stop()
        self._final_render_timer.stop()
        self._execute_render(preview=False)
        self._record_current_state(action_name)

    def trigger_render_immediate(self):
        self._debounce_timer.stop()
        self._final_render_timer.stop()
        self._execute_render(preview=False)

    def _execute_render(self, preview: bool = False):
        params = {
            'global_bypass': self.btn_global_bypass.isChecked(),
            'insert1_enabled': self.insert1_wavelet.is_enabled(),
            'insert2_enabled': self.insert2_rl.is_enabled(),
            'insert3_enabled': self.insert3_tone.is_enabled(),
            'insert4_enabled': self.insert4_mono.is_enabled(),
            'insert5_enabled': self.insert5_lc.is_enabled(),
            'insert6_enabled': self.insert6_color.is_enabled(),
            '_preview': bool(preview),
        }
        params.update(self.insert1_wavelet.get_values())
        params.update(self.insert2_rl.get_values())
        params.update(self.insert3_tone.get_values())
        params.update(self.insert4_mono.get_values())
        params.update(self.insert5_lc.get_values())
        params.update(self.insert6_color.get_values())

        if self._last_dispatched_params == params:
            return
        self._last_dispatched_params = params.copy()
        self._render_start_time = time.perf_counter()
        if hasattr(self, 'lbl_sb_pipeline'):
            self.lbl_sb_pipeline.setText("Processing...")
            self.lbl_sb_pipeline.setStyleSheet("color: #f59e0b; font-weight: bold;")
        self.worker.request_processing(params)

    def _on_frame_rendered(self, processed_frame: np.ndarray):
        if processed_frame is None or processed_frame.size == 0:
            return
        # Игнорируем кадр, если он пришёл от устаревшего расчета другого разрешения
        if self._raw_image is not None:
            if processed_frame.shape[:2] != self._raw_image.shape[:2]:
                return
        render_ms = (time.perf_counter() - getattr(self, '_render_start_time', time.perf_counter())) * 1000.0
        if hasattr(self, 'lbl_sb_pipeline'):
            self.lbl_sb_pipeline.setText(f"Render: {render_ms:.0f} ms")
            self.lbl_sb_pipeline.setStyleSheet("color: #00adb5; font-weight: bold;")
        self._current_processed_image = processed_frame
        self.canvas.set_processed_image(processed_frame)
        if not self._hist_timer.isActive():
            self._hist_timer.start()

    def _sync_histogram_visors(self):
        """Синхронизирует позиции визиров Black Point и White Point с параметрами Insert 3."""
        if hasattr(self, 'insert3_tone') and hasattr(self, 'histogram') and self.histogram is not None:
            vals = self.insert3_tone.get_values()
            ins_enabled = self.insert3_tone.is_enabled()
            black_val = vals.get('tc_black', 0.0)
            black_on = vals.get('tc_black_on', False) and ins_enabled
            gamma_val = vals.get('tc_gamma', 1.0)
            gamma_on = vals.get('tc_gamma_on', False) and ins_enabled
            self.histogram.set_cut_points(black_val, gamma_val, black_on, gamma_on)

    def _flush_histogram_update(self):
        if getattr(self, 'btn_global_bypass', None) and self.btn_global_bypass.isChecked():
            img = self._raw_image
        else:
            img = self._current_processed_image
        if img is None or img.size == 0:
            return
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return
        total_px = h * w
        if total_px > 65536:
            step = max(1, int(np.sqrt(total_px / 65536)))
            sample_img = img[::step, ::step]
        else:
            sample_img = img
        if hasattr(self, 'histogram') and self.histogram is not None:
            self.histogram.update_histogram(sample_img)

    def _on_split_toggled(self, active: bool):
        self.canvas.set_split_view_active(active)
        self.btn_invert_split.setVisible(active)
        self.btn_split_orient.setVisible(active)

    def _on_invert_split_toggled(self, inverted: bool):
        self.canvas.set_split_inverted(inverted)

    def _on_split_orient_toggled(self, checked: bool):
        orient = "Horizontal" if checked else "Vertical"
        self.canvas.set_split_orientation(orient)
        self.btn_split_orient.setToolTip(f"Сплит: {orient} (клик — сменить ориентацию)")

    def _on_global_bypass_toggled(self, checked: bool):
        sound_engine.play_relay_click()
        if checked:
            self.btn_global_bypass.setText("MASTER BYPASS: ACTIVE (RAW VIEW)")
        else:
            self.btn_global_bypass.setText("MASTER BYPASS: INACTIVE (EFFECTS ON)")

        for ins in [
            getattr(self, 'insert1_wavelet', None),
            getattr(self, 'insert2_rl', None),
            getattr(self, 'insert3_tone', None),
            getattr(self, 'insert4_mono', None),
            getattr(self, 'insert5_lc', None),
            getattr(self, 'insert6_color', None),
        ]:
            if ins is not None and hasattr(ins, 'set_master_bypassed'):
                ins.set_master_bypassed(checked)

        if hasattr(self, 'histogram') and hasattr(self.histogram, 'set_master_bypass'):
            self.histogram.set_master_bypass(checked)
        self._flush_histogram_update()

        self.trigger_render_immediate()
        self._record_current_state("Master Bypass")

    def _apply_preset_dict(self, p: dict):
        if 'raw_image' in p and p['raw_image'] is not None:
            if self._raw_image is None or self._raw_image is not p['raw_image']:
                self._raw_image = p['raw_image'].copy()
                self._last_dispatched_params = None
                self.canvas.set_raw_image(self._raw_image)
                self.worker.set_source_image(self._raw_image)
                self.lbl_status.setText(f"{self._current_filename} [{self._raw_image.shape[1]}x{self._raw_image.shape[0]}]")
                self._update_status_image_specs()
                self._update_canvas_guard_state()

        if 'canvas_transform' in p:
            self.canvas.setTransform(p['canvas_transform'])

        inserts = [
            self.insert1_wavelet, self.insert2_rl, self.insert3_tone,
            self.insert4_mono, self.insert5_lc, self.insert6_color
        ]
        for ins in inserts:
            ins.blockSignals(True)

        self.insert1_wavelet.set_enabled(p.get('insert1_enabled', False))
        self.insert1_wavelet.set_values(p)

        self.insert2_rl.set_enabled(p.get('insert2_enabled', False))
        self.insert2_rl.set_values(p)

        self.insert3_tone.set_enabled(p.get('insert3_enabled', False))
        self.insert3_tone.set_values(p)

        self.insert4_mono.set_enabled(p.get('insert4_enabled', False))
        self.insert4_mono.set_values(p)

        self.insert5_lc.set_enabled(p.get('insert5_enabled', False))
        self.insert5_lc.set_values(p)

        self.insert6_color.set_enabled(p.get('insert6_enabled', False))
        self.insert6_color.set_values(p)

        for ins in inserts:
            ins.blockSignals(False)

        self._sync_histogram_visors()

        is_bypass = p.get('global_bypass', False)
        self.btn_global_bypass.blockSignals(True)
        self.btn_global_bypass.setChecked(is_bypass)
        self.btn_global_bypass.setText("MASTER BYPASS: ACTIVE (RAW VIEW)" if is_bypass else "MASTER BYPASS: INACTIVE (EFFECTS ON)")
        self.btn_global_bypass.blockSignals(False)

        for ins in inserts:
            if hasattr(ins, 'set_master_bypassed'):
                ins.set_master_bypassed(is_bypass)

        if hasattr(self, 'histogram') and hasattr(self.histogram, 'set_master_bypass'):
            self.histogram.set_master_bypass(is_bypass)

        self._last_dispatched_params = None
        self.trigger_render_immediate()

    def _populate_presets_combo(self):
        self.combo_presets.blockSignals(True)
        self.combo_presets.clear()
        self.combo_presets.addItems(list(self._builtin_presets.keys()))
        self.combo_presets.insertSeparator(self.combo_presets.count())

        gold_brush = QBrush(QColor("#e5a93b"))

        self.combo_presets.addItem("Save Preset...")
        save_idx = self.combo_presets.count() - 1
        self.combo_presets.setItemData(save_idx, gold_brush, Qt.ItemDataRole.ForegroundRole)
        self.combo_presets.setItemData(save_idx, True, Qt.ItemDataRole.UserRole + 1)

        self.combo_presets.addItem("Load Preset...")
        load_idx = self.combo_presets.count() - 1
        self.combo_presets.setItemData(load_idx, gold_brush, Qt.ItemDataRole.ForegroundRole)
        self.combo_presets.setItemData(load_idx, True, Qt.ItemDataRole.UserRole + 1)

        self.combo_presets.blockSignals(False)

    def _on_preset_selected(self, index: int):
        if index < 0:
            return
        item_text = self.combo_presets.itemText(index)
        if item_text == "Save Preset...":
            self.combo_presets.blockSignals(True)
            self.combo_presets.setCurrentIndex(self._last_preset_index)
            self.combo_presets.blockSignals(False)
            self._save_preset_dialog()
        elif item_text == "Load Preset...":
            self.combo_presets.blockSignals(True)
            self.combo_presets.setCurrentIndex(self._last_preset_index)
            self.combo_presets.blockSignals(False)
            self._load_preset_dialog()
        elif item_text in self._builtin_presets:
            self._last_preset_index = index
            self._apply_preset_dict(self._builtin_presets[item_text])
            self._record_current_state("Preset")

    def _save_preset_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", os.path.join(self._last_save_dir, "my_preset.json"), "JSON Preset (*.json)"
        )
        if not path:
            return

        self._last_save_dir = os.path.dirname(path)
        data = self._get_current_state_dict()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            self.lbl_status.setText(f"Сохранен: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"\u041e\u0448\u0438\u0431\u043a\u0430 \u0437\u0430\u043f\u0438\u0441\u0438 \u0444\u0430\u0439\u043b\u0430:\n{str(e)}")

    def _load_preset_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preset", self._last_open_dir, "JSON Preset (*.json)"
        )
        if not path:
            return

        self._last_open_dir = os.path.dirname(path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._apply_preset_dict(data)
            self._record_current_state("Preset")
            preset_name = f"User: {os.path.splitext(os.path.basename(path))[0]}"
            self._builtin_presets[preset_name] = data
            self._populate_presets_combo()
            new_idx = self.combo_presets.findText(preset_name)
            if new_idx >= 0:
                self._last_preset_index = new_idx
                self.combo_presets.blockSignals(True)
                self.combo_presets.setCurrentIndex(new_idx)
                self.combo_presets.blockSignals(False)
            self.lbl_status.setText(f"Загружен: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"\u041e\u0448\u0438\u0431\u043a\u0430 \u0447\u0442\u0435\u043d\u0438\u044f \u043f\u0440\u0435\u0441\u0435\u0442\u0430:\n{str(e)}")

    def closeEvent(self, event):
        self.worker.cancel_current()
        self.thread.quit()
        self.thread.wait()
        event.accept()