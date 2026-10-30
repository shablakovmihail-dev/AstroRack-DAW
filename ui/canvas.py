#"""Холст AstroCanvas: интерактивный Split View, лупа 300% (Alt), кадрирование и профилометр."""
import os
import time
import math
import logging
import traceback
import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF, QTimer, QPoint, QRect
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsItem, QLabel,
    QFrame, QVBoxLayout, QHBoxLayout, QPushButton, QMenu
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QFont, QPen, QBrush,
    QPainterPath, QDragEnterEvent, QDropEvent, QDragLeaveEvent, QDragMoveEvent,
    QRadialGradient, QLinearGradient
)
from .widgets import THEMES
from .measurement_hud import MeasurementHUD, MeasurementBottomBar
from core.geometry_calc import (
    circle_from_3_points,
    fit_circle_3points,
    calculate_spherical_correction,
    calculate_crater_ellipse
)
from utils.file_io import get_resource_path


def ensure_samples_dir() -> str:
    """Проверяет наличие папки samples/ и создает демонстрационные астроснимки при необходимости."""
    samples_dir = get_resource_path("samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir, exist_ok=True)

    # Проверка наличия хотя бы одного графического файла в папке или ее поддиректориях
    image_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.fit', '.fits')
    has_content = False
    for root, dirs, files in os.walk(samples_dir):
        if any(f.lower().endswith(image_exts) for f in files):
            has_content = True
            break

    if not has_content:
        _create_fallback_samples(samples_dir)

    return samples_dir


def _create_fallback_samples(samples_dir: str):
    """Генерация базовых синтетических астроснимков, если папка пуста."""
    try:
        size = 500
        y, x = np.ogrid[:size, :size]
        center = size / 2.0
        r = np.hypot(x - center, y - center)
        planet_mask = r <= 180

        # 1. Jupiter (RGB Stacking)
        jup = np.zeros((size, size, 3), dtype=np.uint8)
        bands = np.sin((y - center) * 0.12) * 35.0 + np.sin((y - center) * 0.05) * 45.0
        limb_darkening = np.clip(1.0 - (r / 180.0) ** 2, 0.0, 1.0) ** 0.4

        r_ch = np.clip((180 + bands * 0.8) * limb_darkening, 0, 255).astype(np.uint8)
        g_ch = np.clip((140 + bands * 0.5) * limb_darkening, 0, 255).astype(np.uint8)
        b_ch = np.clip((100 + bands * 0.2) * limb_darkening, 0, 255).astype(np.uint8)

        jup[planet_mask, 0] = b_ch[planet_mask]
        jup[planet_mask, 1] = g_ch[planet_mask]
        jup[planet_mask, 2] = r_ch[planet_mask]
        cv2.imwrite(os.path.join(samples_dir, "Jupiter (RGB Stacking).png"), jup)

        # 2. Moon Craters (High Contrast)
        noise = np.random.normal(128, 25, (size, size))
        moon = np.clip(noise, 0, 255).astype(np.uint8)
        for cx, cy, cr in [(200, 200, 50), (350, 280, 70), (160, 360, 35), (320, 120, 40)]:
            cr_dist = np.hypot(x - cx, y - cy)
            rim = np.exp(-((cr_dist - cr) ** 2) / 30.0) * 80.0
            floor = np.clip(1.0 - cr_dist / cr, 0, 1) * -60.0
            moon = np.clip(moon + rim + floor, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(samples_dir, "Moon Craters (High Contrast).png"), moon)

        # 3. Orion Core (Deep-Sky)
        orion = np.zeros((size, size, 3), dtype=np.uint8)
        d_center = np.hypot(x - 220, y - 240)
        nebula_core = np.exp(-(d_center ** 2) / (2 * (80.0 ** 2))) * 220.0
        orion[:, :, 0] = np.clip(nebula_core * 0.9, 0, 255).astype(np.uint8)
        orion[:, :, 1] = np.clip(nebula_core * 0.5, 0, 255).astype(np.uint8)
        orion[:, :, 2] = np.clip(nebula_core * 1.0, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(samples_dir, "Orion Core (Deep-Sky).png"), orion)
    except Exception:
        pass


class ZeroStateWidget(QFrame):
    """Интерактивный оверлей пустого холста (Zero State / Welcome Canvas) с Drag & Drop и демо-библиотекой."""
    fileSelected = pyqtSignal(str)
    openFileDialogRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ZeroStateWidget")
        self.setAcceptDrops(True)
        self._is_drag_active = False

        self._normal_style = """
            QFrame#ZeroStateWidget {
                background-color: rgba(20, 23, 33, 0.92);
                border: 2px dashed #334155;
                border-radius: 8px;
            }
        """
        self._hover_style = """
            QFrame#ZeroStateWidget {
                background-color: rgba(20, 56, 60, 0.94);
                border: 2px dashed #00fff2;
                border-radius: 8px;
            }
        """
        self.setStyleSheet(self._normal_style)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 26, 36, 24)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 1. Фирменный полигональный логотип
        self.lbl_icon = QLabel()
        self.lbl_icon.setFixedSize(96, 96)
        self.lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_icon.setStyleSheet("background: transparent;")
        icon_path = get_resource_path(os.path.join("assets", "icons", "icon_128x128.png"))
        if os.path.isfile(icon_path):
            self.lbl_icon.setPixmap(QPixmap(icon_path).scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.lbl_icon.setText("🔭")
            self.lbl_icon.setStyleSheet("font-size: 38px; color: #00adb5; background: transparent;")
        layout.addWidget(self.lbl_icon, 0, Qt.AlignmentFlag.AlignCenter)

        # 2. Крупный заголовок
        lbl_title = QLabel("Перетащите снимок сюда (Drag & Drop)")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("""
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 15px;
            font-weight: 700;
            color: #f1f5f9;
            background: transparent;
        """)
        layout.addWidget(lbl_title)

        # 3. Подпись форматов
        lbl_sub = QLabel("Поддерживаются TIFF, FITS, PNG, JPG (16-bit / 32-bit float)")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_sub.setStyleSheet("""
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11px;
            color: #94a3b8;
            background: transparent;
        """)
        layout.addWidget(lbl_sub)
        layout.addSpacing(8)

        # 4. Ряд кнопок
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_browse = QPushButton("Выбрать файл...")
        self.btn_browse.setFixedHeight(34)
        self.btn_browse.setStyleSheet("""
            QPushButton {
                background-color: #1e2433;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #cbd5e1;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #273044;
                border-color: #00adb5;
                color: #ffffff;
            }
        """)
        self.btn_browse.clicked.connect(self.openFileDialogRequested.emit)
        btn_layout.addWidget(self.btn_browse)

        self.btn_demo = QPushButton("🗁  Загрузить демо-снимок ▾")
        self.btn_demo.setFixedHeight(34)
        self.btn_demo.setStyleSheet("""
            QPushButton {
                background-color: #1e2433;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #94a3b8;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #273044;
                border-color: #00adb5;
                color: #ffffff;
            }
        """)
        self.btn_demo.clicked.connect(self._on_demo_clicked)
        btn_layout.addWidget(self.btn_demo)

        layout.addLayout(btn_layout)

    def apply_theme(self, theme: dict):
        bright = theme.get('bright', '#00fff2')
        self._hover_style = f"""
            QFrame#ZeroStateWidget {{
                background-color: rgba(20, 56, 60, 0.94);
                border: 2px dashed {bright};
                border-radius: 8px;
            }}
        """
        if self._is_drag_active:
            self.setStyleSheet(self._hover_style)
        self.lbl_icon.setStyleSheet("background: transparent;")
        self.btn_demo.setStyleSheet(f"""
            QPushButton {{
                background-color: #1e2433;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #94a3b8;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                font-weight: 600;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background-color: #273044;
                border-color: {bright};
                color: #ffffff;
            }}
        """)

    def set_drag_active(self, active: bool):
        if self._is_drag_active != active:
            self._is_drag_active = active
            self.setStyleSheet(self._hover_style if active else self._normal_style)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.set_drag_active(True)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.set_drag_active(False)

    def dropEvent(self, event):
        self.set_drag_active(False)
        for url in event.mimeData().urls():
            fp = url.toLocalFile()
            if os.path.isfile(fp):
                self.fileSelected.emit(fp)
                event.acceptProposedAction()
                break

    def _on_demo_clicked(self):
        menu = self._build_samples_menu()
        if menu is not None and not menu.isEmpty():
            pos = self.btn_demo.mapToGlobal(QPoint(0, self.btn_demo.height() + 4))
            menu.exec(pos)

    def _build_samples_menu(self) -> QMenu | None:
        samples_dir = ensure_samples_dir()
        if not os.path.exists(samples_dir):
            return None

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #141722;
                border: 1px solid #334155;
                border-radius: 6px;
                color: #e2e8f0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px 6px 12px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #00adb5;
                color: #0f1117;
                font-weight: bold;
            }
            QMenu::separator {
                height: 1px;
                background-color: #242b3d;
                margin: 4px 6px;
            }
        """)

        image_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.fit', '.fits')

        root_files = [
            f for f in sorted(os.listdir(samples_dir))
            if os.path.isfile(os.path.join(samples_dir, f)) and f.lower().endswith(image_exts)
        ]
        for rf in root_files:
            name = os.path.splitext(rf)[0]
            full_p = os.path.join(samples_dir, rf)
            act = menu.addAction(f"★  {name}")
            act.triggered.connect(lambda checked=False, p=full_p: self.fileSelected.emit(p))

        cat_icons = {
            "MOON": "🌙  Moon",
            "JUPITER": "♃  Jupiter",
            "MARS": "♂  Mars",
            "SATURN": "♄  Saturn",
            "DEEP_SKY": "🌌  Deep-Sky",
            "SUN": "☉  Sun",
            "VENUS": "♀  Venus",
            "MERCURY": "☿  Mercury",
            "URANUS": "♅  Uranus",
            "NEPTUNE": "♆  Neptune"
        }

        subdirs = [
            d for d in sorted(os.listdir(samples_dir))
            if os.path.isdir(os.path.join(samples_dir, d))
        ]

        if root_files and subdirs:
            menu.addSeparator()

        for d in subdirs:
            dir_path = os.path.join(samples_dir, d)
            sub_files = [
                f for f in sorted(os.listdir(dir_path))
                if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith(image_exts)
            ]
            if not sub_files:
                continue
            title = cat_icons.get(d.upper(), f"📁  {d}")
            sub_menu = menu.addMenu(f"{title} ({len(sub_files)})")
            sub_menu.setStyleSheet(menu.styleSheet())
            for sf in sub_files:
                sf_name = os.path.splitext(sf)[0]
                full_sf = os.path.join(dir_path, sf)
                act = sub_menu.addAction(sf_name)
                act.triggered.connect(lambda checked=False, p=full_sf: self.fileSelected.emit(p))

        return menu


class CanvasStatusOverlay(QLabel):
    """Плавающий полупрозрачный оверлей статуса в левом нижнем углу холста."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            background-color: rgba(18, 18, 26, 215);
            border: 1px solid #323244;
            border-radius: 6px;
            color: #00adb5;
            font-size: 11px;
            font-weight: bold;
            padding: 4px 10px;
        """)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)


class CropOverlayItem(QGraphicsItem):
    """Интерактивная рамка кадрирования с затемнением внешних полей и 4 угловыми ручками."""

    def __init__(self, img_w: float, img_h: float):
        super().__init__()
        self.img_w = max(1.0, img_w)
        self.img_h = max(1.0, img_h)
        margin_x = self.img_w * 0.1
        margin_y = self.img_h * 0.1
        self.crop_rect = QRectF(margin_x, margin_y, self.img_w * 0.8, self.img_h * 0.8)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.img_w, self.img_h)

    def paint(self, painter: QPainter, option, widget=None):
        painter.save()
        path = QPainterPath()
        path.addRect(QRectF(0, 0, self.img_w, self.img_h))
        path.addRect(self.crop_rect)
        painter.fillPath(path, QBrush(QColor(0, 0, 0, 160)))

        border_pen = QPen(QColor("#00fff2"), 1.5, Qt.PenStyle.DashLine)
        border_pen.setCosmetic(True)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.crop_rect)

        r = self.crop_rect
        view_scale = max(1e-4, painter.transform().m11())

        # Оптические направляющие центрирования (Precision Centering Guides)
        # 1. Диагональные пунктирные направляющие (X-образное перекрестие)
        guide_pen = QPen(QColor(0, 229, 255, 115), 1.0, Qt.PenStyle.DashLine)
        guide_pen.setCosmetic(True)
        painter.setPen(guide_pen)
        painter.drawLine(QPointF(r.left(), r.top()), QPointF(r.right(), r.bottom()))
        painter.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.top()))

        # 2. Оптическое перекрестие / прицел (bullseye reticle) в точке пересечения диагоналей
        c = r.center()
        cx, cy = c.x(), c.y()
        reticle_pen = QPen(QColor(0, 229, 255, 140), 1.0, Qt.PenStyle.SolidLine)
        reticle_pen.setCosmetic(True)
        painter.setPen(reticle_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        r_circle = 4.5 / view_scale
        cross_half = 5.5 / view_scale

        painter.drawEllipse(c, r_circle, r_circle)
        painter.drawLine(QPointF(cx - cross_half, cy), QPointF(cx + cross_half, cy))
        painter.drawLine(QPointF(cx, cy - cross_half), QPointF(cx, cy + cross_half))

        # Угловые маркеры-манипуляторы
        corners = [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]
        handle_pen = QPen(QColor("#00fff2"), 1.5)
        handle_pen.setCosmetic(True)
        painter.setPen(handle_pen)
        painter.setBrush(QBrush(QColor("#1a1a24")))

        handle_scene_half = (10.0 / view_scale) / 2.0
        handle_scene_size = handle_scene_half * 2.0

        for c_corner in corners:
            painter.drawRect(QRectF(c_corner.x() - handle_scene_half, c_corner.y() - handle_scene_half, handle_scene_size, handle_scene_size))

        painter.restore()


class SplitImageGraphicsItem(QGraphicsItem):
    """Элемент сцены, отображающий две половинки изображения (Before и After)."""

    def __init__(self):
        super().__init__()
        self.raw_pixmap = QPixmap()
        self.processed_pixmap = QPixmap()
        self.split_active = False
        self.split_fraction = 0.5
        self.split_orientation = "Vertical"
        self.split_inverted = False

    def boundingRect(self) -> QRectF:
        w = max(self.raw_pixmap.width(), self.processed_pixmap.width())
        h = max(self.raw_pixmap.height(), self.processed_pixmap.height())
        return QRectF(0, 0, max(w, 0), max(h, 0))

    def paint(self, painter: QPainter, option, widget=None):
        if self.processed_pixmap.isNull():
            if not self.raw_pixmap.isNull():
                painter.drawPixmap(0, 0, self.raw_pixmap)
            return

        w = self.processed_pixmap.width()
        h = self.processed_pixmap.height()
        if w <= 0 or h <= 0:
            return

        if not self.split_active or self.raw_pixmap.isNull():
            painter.drawPixmap(0, 0, self.processed_pixmap)
            return

        # Защита от рассинхронизации размеров raw и processed при смене изображения
        if self.raw_pixmap.width() != w or self.raw_pixmap.height() != h:
            painter.drawPixmap(0, 0, self.processed_pixmap)
            return

        pix_first = self.processed_pixmap if self.split_inverted else self.raw_pixmap
        pix_second = self.raw_pixmap if self.split_inverted else self.processed_pixmap

        # Аппаратное разделение слоёв через QPainter в экранных координатах виджета
        painter.save()
        # 1. Отрисовка полного изображения Before с текущими параметрами зума (zoom_factor) и смещения (pan_offset)
        painter.drawPixmap(0, 0, pix_first)

        # 2. Определение области отсечения (clipping rect) для слоя After по координате сплиттера в координатах виджета
        orig_transform = painter.transform()
        vp_w = widget.width() if widget is not None else (painter.device().width() if painter.device() is not None else int(w))
        vp_h = widget.height() if widget is not None else (painter.device().height() if painter.device() is not None else int(h))

        painter.save()
        painter.resetTransform()

        if self.split_orientation == "Vertical":
            split_pt = orig_transform.map(QPointF(w * self.split_fraction, 0.0))
            split_pos_x = int(round(split_pt.x()))
            clip_rect = QRect(split_pos_x, 0, max(0, vp_w - split_pos_x), vp_h)
            painter.setClipRect(clip_rect)
        else:
            split_pt = orig_transform.map(QPointF(0.0, h * self.split_fraction))
            split_pos_y = int(round(split_pt.y()))
            clip_rect = QRect(0, split_pos_y, vp_w, max(0, vp_h - split_pos_y))
            painter.setClipRect(clip_rect)

        # 3. Поверх отрисовываем полное изображение After с абсолютно теми же трансформациями zoom_factor и pan_offset
        painter.setTransform(orig_transform)
        painter.drawPixmap(0, 0, pix_second)
        painter.restore()

        painter.restore()


class AstroCanvas(QGraphicsView):
    """Интерактивный вьюпорт с зумом, косметической линией сплита, лупой и профилометром."""
    cropApplied = pyqtSignal(QRectF)
    cropCancelled = pyqtSignal()
    profileLineSelected = pyqtSignal(QPointF, QPointF)
    zoomChanged = pyqtSignal(str)
    psfPointClicked = pyqtSignal(QPointF)
    rulerMeasured = pyqtSignal(float, float, str)
    rulerSubmodeChanged = pyqtSignal(str)
    fileDropped = pyqtSignal(str)
    openFileDialogRequested = pyqtSignal()
    pixelProbeChanged = pyqtSignal(str, str)
    clipping_overlay_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._item = SplitImageGraphicsItem()
        self._scene.addItem(self._item)
        self._current_frame_float = None

        # Интерактивная подсветка клиппинга (Sonar Ping Overlay)
        self.clipping_overlay_mode = None  # None | "shadows" | "highlights"
        self._clipping_mask_pixmap = None
        self._clipping_centroids = []
        self._radar_timer = QTimer(self)
        self._radar_timer.setInterval(25)
        self._radar_timer.timeout.connect(self._on_radar_timer_tick)
        self._radar_start_time = 0.0

        self._crop_item = None
        self._crop_mode = False
        self._crop_drag_mode = None
        self._crop_drag_start = QPointF()
        self._crop_rect_start = QRectF()

        self._dragging_split = False

        self._alt_pressed = False
        self._loupe_mode = False
        self._loupe_zoom = 2.0
        self._mouse_vp = QPointF()
        self._theme = THEMES['A']

        self._profile_mode = False
        self._profile_p1 = None
        self._profile_p2 = None
        self._profile_dragging = False

        self._psf_mode = False
        self._psf_last_pos = None

        self._ruler_mode = False
        self._ruler_p1 = None
        self._ruler_p2 = None
        self._ruler_dragging = False
        self._telescope_focal_len = 1000.0
        self._telescope_pixel_size = 3.76
        self._lunar_distance_km = 384400.0
        self._ruler_unit = "all"

        # Служебные структуры для диска Луны и эллипса кратера (с защитой состояния)
        self.disk_points = []
        self.three_points = self.disk_points
        self.three_points_result = None
        self.disk_center = None
        self.disk_radius = 0.0
        self.is_drawing_disk = False
        self.ellipse_center = None
        self.ellipse_a = 0.0
        self.ellipse_b = 0.0
        self.ellipse_angle = 0.0
        self.ellipse_stage = 0  # 0: idle, 1: defining primary axis, 2: defining secondary axis
        self.ellipse_start = None
        self.ellipse_end = None
        self.active_mode = "ruler"  # "ruler", "moon_disk", "crater_ellipse"

        # Сферическая привязка диска Луны и эллиптический замер кратеров
        self._meas_submode = "ruler"  # "ruler" | "disk" | "ellipse"
        self._disk_method = "center_radius"  # "center_radius" | "three_points"
        self.calibrated_disk = None  # Tuple[float, float, float] (x0, y0, radius) или None

        # Bounding Box привязки диска Луны (1:1 Drag from Corner)
        self.disk_drag_start = None     # QPointF в координатах картинки
        self.disk_drag_current = None   # QPointF в координатах картинки
        self.disk_is_dragging = False   # зажата ли мышь
        self._disk_rect_img = None      # QRectF рамки строго в координатах изображения
        self.temp_disk_rect = None      # псевдоним для обратной совместимости
        self.pending_disk = None        # (cx, cy, r)
        self._disk_transform_mode = None  # "draw" | "move" | "resize" | None
        self._disk_drag_start_pos = None
        self._disk_drag_start_center = None
        self._disk_drag_start_radius = 0.0

        # Совместимость структур
        self._disk_draft_center = None
        self._disk_draft_radius = 0.0
        self._disk_draft_dragging = False
        self._disk_draft_drag_target = None  # "center" | "radius" | None
        self._disk_points = self.disk_points
        self._disk_state = "idle"
        self._disk_draw_start = None
        self._disk_active_handle = None
        self._disk_drag_start_mouse = None
        self._drag_start_center = None
        self._drag_start_radius = 0.0
        self._disk_anchor = None
        self._disk_hover_handle = None

        # Интерактивное состояние замера кратера (Two-stage Drag)
        self._ellipse_step = 0  # 0: idle, 1: dragging major axis, 2: adjusting minor axis, 3: completed
        self._ellipse_p_start = None
        self._ellipse_p_end = None
        self._ellipse_cursor = None
        self._ellipse_data = None

        # Предвыделенные буферы для конвертации numpy -> QImage -> QPixmap без лишних копий
        self._raw_u8_buf = None
        self._raw_qimg = None
        self._proc_u8_buf = None
        self._proc_qimg = None
        self._last_scene_size = (0, 0)

        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(350)
        self._zoom_timer.timeout.connect(lambda: self.zoomChanged.emit("Zoom"))

        self.setBackgroundBrush(QColor("#0f0f12"))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

        self.status_overlay = CanvasStatusOverlay(self.viewport())
        self.status_overlay.setText("No image loaded")
        self.status_overlay.adjustSize()
        self.status_overlay.hide()

        self.measurement_hud = MeasurementBottomBar()
        self.measurement_hud.hide()
        self.measurement_shelf = self.measurement_hud
        self.measurement_hud.mode_changed.connect(self.set_measurement_submode)
        self.measurement_hud.disk_method_changed.connect(self.set_disk_method)
        self.measurement_hud.apply_alignment_requested.connect(self.apply_disk_alignment)
        self.measurement_hud.reset_disk_requested.connect(self.reset_disk_alignment)

        self.welcome_widget = ZeroStateWidget(self.viewport())
        self.welcome_widget.fileSelected.connect(self.fileDropped.emit)
        self.welcome_widget.openFileDialogRequested.connect(self.openFileDialogRequested.emit)
        self.welcome_widget.show()

    @staticmethod
    def _convert_float_to_u8_inplace(arr: np.ndarray, dst_buf: np.ndarray | None) -> np.ndarray:
        """Однопроходная конвертация float32 [0..1] -> C-contiguous uint8 в переиспользуемый буфер."""
        if arr is None or arr.size == 0:
            return np.empty((0, 0), dtype=np.uint8)
        if arr.dtype == np.uint8:
            if arr.flags['C_CONTIGUOUS']:
                return arr
            if dst_buf is None or dst_buf.shape != arr.shape:
                dst_buf = np.empty(arr.shape, dtype=np.uint8, order='C')
            np.copyto(dst_buf, arr)
            return dst_buf

        src = arr if arr.flags['C_CONTIGUOUS'] else np.ascontiguousarray(arr, dtype=np.float32)
        if dst_buf is None or dst_buf.shape != src.shape or not dst_buf.flags['C_CONTIGUOUS']:
            dst_buf = np.empty(src.shape, dtype=np.uint8, order='C')

        # SIMD-конвертация с насыщением [0..255] без промежуточных float-массивов
        cv2.addWeighted(src, 255.0, src, 0.0, 0.0, dst=dst_buf, dtype=cv2.CV_8U)
        return dst_buf

    @staticmethod
    def _wrap_u8_as_qimage(img_u8: np.ndarray) -> QImage:
        """Оборачивает C-contiguous uint8 буфер в QImage без копирования памяти."""
        if img_u8 is None or img_u8.size == 0:
            return QImage()
        if img_u8.ndim == 2:
            h, w = img_u8.shape
            return QImage(img_u8.data, w, h, w, QImage.Format.Format_Grayscale8)
        h, w, _ = img_u8.shape
        return QImage(img_u8.data, w, h, 3 * w, QImage.Format.Format_RGB888)

    @classmethod
    def _numpy_to_pixmap(cls, arr: np.ndarray) -> QPixmap:
        if arr is None or arr.size == 0:
            return QPixmap()
        img_u8 = cls._convert_float_to_u8_inplace(arr, None)
        qimg = cls._wrap_u8_as_qimage(img_u8)
        return QPixmap.fromImage(qimg, Qt.ImageConversionFlag.NoFormatConversion)

    def _sync_scene_bounds_if_changed(self):
        rect = self._item.boundingRect()
        new_size = (int(rect.width()), int(rect.height()))
        if new_size != self._last_scene_size:
            self._last_scene_size = new_size
            self._item.prepareGeometryChange()
            self._scene.setSceneRect(rect)
            self._sync_crop_overlay_size()

    def _on_radar_timer_tick(self):
        if self.clipping_overlay_mode is not None:
            self.viewport().update()

    def get_clipped_overlay_pixmap(self, mode="peaks") -> Optional[QPixmap]:
        """Генерирует полупрозрачный слой маски клиппинга без вылетов при любых форматах (uint8, uint16, float, Mono, RGB)."""
        try:
            img = self._current_frame_float
            if img is None or img.size == 0:
                return None

            h, w = img.shape[:2]

            # 1. Приведение к единому каналу яркости (Luminance) [0.0, 1.0]:
            if img.ndim == 3 and img.shape[2] >= 3:
                # Веса sRGB для честной яркости Rec.709:
                r = img[:, :, 0].astype(np.float32)
                g = img[:, :, 1].astype(np.float32)
                b = img[:, :, 2].astype(np.float32)
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            else:
                mono = img[:, :, 0] if (img.ndim == 3 and img.shape[2] >= 1) else img
                lum = mono.astype(np.float32)

            # Нормализация к диапазону [0.0, 1.0], если данные в [0, 255] или [0, 65535]:
            if img.dtype == np.uint8 or lum.max() > 255.0:
                lum = lum / 255.0
            elif img.dtype == np.uint16 or lum.max() > 1.0:
                lum = lum / 65535.0 if lum.max() > 255.0 else lum / 255.0
            else:
                lum = np.clip(lum, 0.0, 1.0)

            # 2. Построение булевой маски:
            if mode in ("peaks", "highlights"):
                # Пережжённые пиксели (порог >= 0.999):
                mask = lum >= 0.999
                color_rgba = (255, 40, 40, 220)  # Ярко-красный полупрозрачный (ARGB: A=220, R=255, G=40, B=40)
            else:
                # Проваленные тени (порог <= 0.001):
                mask = lum <= 0.001
                color_rgba = (0, 200, 255, 220)  # Неоново-голубой (ARGB: A=220, R=0, G=200, B=255)

            if not np.any(mask):
                return None  # Нет выбитых пикселей — не тратим память

            # 3. Формирование 4-канального RGBA изображения (uint8):
            overlay_array = np.zeros((h, w, 4), dtype=np.uint8)
            # В формате ARGB32_Premultiplied порядок байт на little-endian: B, G, R, A
            # Для надежности задаем через прямое присвоение BGRA
            r_val, g_val, b_val, a_val = color_rgba[0], color_rgba[1], color_rgba[2], color_rgba[3]
            alpha_f = a_val / 255.0
            premul_r = int(round(r_val * alpha_f))
            premul_g = int(round(g_val * alpha_f))
            premul_b = int(round(b_val * alpha_f))

            overlay_array[mask] = (premul_b, premul_g, premul_r, a_val)
            overlay_array = np.ascontiguousarray(overlay_array)

            # 4. Безопасное создание QImage:
            qimg = QImage(
                overlay_array.data,
                w,
                h,
                w * 4,
                QImage.Format.Format_ARGB32_Premultiplied
            )
            return QPixmap.fromImage(qimg.copy())  # .copy() обязателен, чтобы отвязать память от NumPy

        except Exception as e:
            logging.error(f"Error generating clip overlay: {e}", exc_info=True)
            return None

    def set_clipping_overlay_mode(self, mode: Optional[str]):
        """Устанавливает или отключает интерактивную подсветку клиппинга ('shadows', 'highlights', None)."""
        if mode not in ("shadows", "highlights", None):
            mode = None

        self.clipping_overlay_mode = mode

        if mode is None:
            if hasattr(self, '_radar_timer') and self._radar_timer.isActive():
                self._radar_timer.stop()
            self._clipping_mask_pixmap = None
            self._clipping_centroids = []
            self.clipping_overlay_changed.emit(None)
            self.viewport().update()
            return

        img = self._current_frame_float
        if img is None or img.size == 0:
            self._clipping_mask_pixmap = None
            self._clipping_centroids = []
            self.clipping_overlay_changed.emit(mode)
            self.viewport().update()
            return

        try:
            # Безопасная генерация оверлея
            self._clipping_mask_pixmap = self.get_clipped_overlay_pixmap(mode)

            # Вычисление центроидов для анимации радара
            if self._clipping_mask_pixmap is not None:
                # Получаем маску для расчета связных компонент
                h, w = img.shape[:2]
                if img.ndim == 3 and img.shape[2] >= 3:
                    lum = 0.2126 * img[:, :, 0].astype(np.float32) + 0.7152 * img[:, :, 1].astype(np.float32) + 0.0722 * img[:, :, 2].astype(np.float32)
                else:
                    mono = img[:, :, 0] if (img.ndim == 3 and img.shape[2] >= 1) else img
                    lum = mono.astype(np.float32)

                if img.dtype == np.uint8 or lum.max() > 255.0:
                    lum = lum / 255.0
                elif img.dtype == np.uint16 or lum.max() > 1.0:
                    lum = lum / 65535.0 if lum.max() > 255.0 else lum / 255.0

                if mode == "highlights":
                    mask_calc = (lum >= 0.999)
                else:
                    mask_calc = (lum <= 0.001)

                try:
                    mask_u8 = np.ascontiguousarray(mask_calc.astype(np.uint8))
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        mask_u8, connectivity=8
                    )
                    if num_labels > 1:
                        areas = stats[1:, cv2.CC_STAT_AREA]
                        top_idxs = np.argsort(areas)[::-1][:8]
                        self._clipping_centroids = [
                            QPointF(float(centroids[1 + i][0]), float(centroids[1 + i][1]))
                            for i in top_idxs
                        ]
                    else:
                        pts = np.argwhere(mask_calc)
                        if len(pts) > 0:
                            my, mx = np.mean(pts, axis=0)
                            self._clipping_centroids = [QPointF(float(mx), float(my))]
                        else:
                            self._clipping_centroids = []
                except Exception:
                    pts = np.argwhere(mask_calc)
                    if len(pts) > 0:
                        my, mx = np.mean(pts, axis=0)
                        self._clipping_centroids = [QPointF(float(mx), float(my))]
                    else:
                        self._clipping_centroids = []
            else:
                self._clipping_centroids = []
        except Exception as e:
            logging.error(f"Error setting clipping overlay mode: {e}", exc_info=True)
            self._clipping_mask_pixmap = None
            self._clipping_centroids = []

        self._radar_start_time = time.monotonic()
        if not self._radar_timer.isActive() and len(self._clipping_centroids) > 0:
            self._radar_timer.start(25)

        self.clipping_overlay_changed.emit(mode)
        self.viewport().update()

    def toggle_clipping_overlay(self, mode: str):
        """Переключает режим подсветки (при повторном клике отключает)."""
        if self.clipping_overlay_mode == mode:
            self.set_clipping_overlay_mode(None)
        else:
            self.set_clipping_overlay_mode(mode)

    def reset_for_new_image(self):
        """Полный сброс состояния холста, оверлеев и буферов перед загрузкой нового файла."""
        self.set_clipping_overlay_mode(None)
        self.set_split_view_active(False)
        self.clear_ruler()
        self.clear_psf_marker()
        self.reset_to_default_tool()
        self._current_frame_float = None
        self.pixelProbeChanged.emit("X: ---, Y: ---", "")
        self._raw_u8_buf = None
        self._raw_qimg = None
        self._proc_u8_buf = None
        self._proc_qimg = None
        self._last_scene_size = (0, 0)
        self._item.raw_pixmap = QPixmap()
        self._item.processed_pixmap = QPixmap()
        self._item.prepareGeometryChange()
        self._scene.setSceneRect(QRectF(0, 0, 0, 0))
        self.viewport().update()

    def set_raw_image(self, img_float: np.ndarray):
        if img_float is None or img_float.size == 0:
            return
        self._current_frame_float = img_float
        self._raw_u8_buf = self._convert_float_to_u8_inplace(img_float, self._raw_u8_buf)
        self._raw_qimg = self._wrap_u8_as_qimage(self._raw_u8_buf)
        self._item.raw_pixmap = QPixmap.fromImage(
            self._raw_qimg, Qt.ImageConversionFlag.NoFormatConversion
        )
        # Синхронизируем processed_pixmap, чтобы исключить рассинхронизацию размеров
        if self._item.processed_pixmap.isNull() or self._item.processed_pixmap.size() != self._item.raw_pixmap.size():
            self._item.processed_pixmap = self._item.raw_pixmap.copy()
        self._sync_scene_bounds_if_changed()
        self.fit_current_image()
        if hasattr(self, 'welcome_widget'):
            self.welcome_widget.setVisible(False)
        if hasattr(self, 'status_overlay'):
            self.status_overlay.setVisible(True)
        self.zoomChanged.emit(self.get_current_zoom_str())
        if getattr(self, 'clipping_overlay_mode', None) is not None:
            self.set_clipping_overlay_mode(self.clipping_overlay_mode)
        self.viewport().update()

    def set_processed_image(self, img_float: np.ndarray):
        if img_float is None or img_float.size == 0:
            return
        self._current_frame_float = img_float
        is_first = self._item.processed_pixmap.isNull()
        self._proc_u8_buf = self._convert_float_to_u8_inplace(img_float, self._proc_u8_buf)
        self._proc_qimg = self._wrap_u8_as_qimage(self._proc_u8_buf)
        self._item.processed_pixmap = QPixmap.fromImage(
            self._proc_qimg, Qt.ImageConversionFlag.NoFormatConversion
        )
        self._sync_scene_bounds_if_changed()
        if is_first:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.zoomChanged.emit(self.get_current_zoom_str())
        if hasattr(self, 'welcome_widget'):
            self.welcome_widget.setVisible(False)
        if getattr(self, 'clipping_overlay_mode', None) is not None:
            self.set_clipping_overlay_mode(self.clipping_overlay_mode)
        self.viewport().update()

    def show_welcome_state(self):
        """Переводит холст в интерактивный режим ожидания (Zero State)."""
        self.reset_for_new_image()
        if hasattr(self, 'welcome_widget'):
            self.welcome_widget.setVisible(True)
            self._update_welcome_pos()
        if hasattr(self, 'status_overlay'):
            self.status_overlay.setVisible(False)
        self.viewport().update()

    def apply_theme(self, theme: dict):
        self._theme = theme
        if hasattr(self, 'welcome_widget') and self.welcome_widget is not None:
            self.welcome_widget.apply_theme(theme)
        self.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_overlay_pos()
        self._update_welcome_pos()
        self._update_measurement_hud_pos()

    def _update_measurement_hud_pos(self):
        # Панель измерений зафиксирована в нижней части контейнера под холстом
        pass

    def _update_overlay_pos(self):
        if hasattr(self, 'status_overlay'):
            self.status_overlay.adjustSize()
            h = self.status_overlay.height()
            self.status_overlay.move(12, max(0, self.viewport().height() - h - 12))

    def _update_welcome_pos(self):
        if hasattr(self, 'welcome_widget') and self.welcome_widget.isVisible():
            vp_w = self.viewport().width()
            vp_h = self.viewport().height()
            w = min(480, max(360, vp_w - 40))
            h = min(310, max(240, vp_h - 40))
            self.welcome_widget.resize(w, h)
            x = max(0, (vp_w - w) // 2)
            y = max(0, (vp_h - h) // 2)
            self.welcome_widget.move(x, y)

    def set_status_text(self, text: str):
        if hasattr(self, 'status_overlay'):
            self.status_overlay.setText(text)
            self._update_overlay_pos()

    def get_current_zoom_pct(self) -> int:
        return int(round(self.transform().m11() * 100))

    def get_current_zoom_str(self) -> str:
        return f"Zoom: {self.get_current_zoom_pct()}%"

    def fit_current_image(self):
        self._scene.setSceneRect(self._item.boundingRect())
        if not self._item.boundingRect().isEmpty():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.get_current_zoom_str())
        self.viewport().update()

    def _emit_pixel_probe(self, scene_pos: QPointF):
        if self._current_frame_float is None or self._current_frame_float.size == 0:
            self.pixelProbeChanged.emit("X: ---, Y: ---", "")
            return

        img = self._current_frame_float
        h, w = img.shape[:2]
        x = int(np.floor(scene_pos.x()))
        y = int(np.floor(scene_pos.y()))

        if 0 <= x < w and 0 <= y < h:
            coords_str = f"X: {x}, Y: {y}"
            if img.ndim == 2:
                val = img[y, x]
                if img.dtype == np.uint8:
                    val_str = f"L: {int(val)}"
                else:
                    val_str = f"L: {float(val):.3f}"
            elif img.ndim == 3 and img.shape[2] >= 3:
                r, g, b = img[y, x, 0], img[y, x, 1], img[y, x, 2]
                if img.dtype == np.uint8:
                    val_str = f"RGB: [{int(r)}, {int(g)}, {int(b)}]"
                else:
                    val_str = f"RGB: [{float(r):.2f}, {float(g):.2f}, {float(b):.2f}]"
            else:
                val_str = ""
            self.pixelProbeChanged.emit(coords_str, val_str)
        else:
            self.pixelProbeChanged.emit("X: ---, Y: ---", "")

    def _sync_crop_overlay_size(self):
        if self._crop_item is None:
            return
        w = self._item.boundingRect().width()
        h = self._item.boundingRect().height()
        if abs(self._crop_item.img_w - w) < 0.5 and abs(self._crop_item.img_h - h) < 0.5:
            return
        self._crop_item.prepareGeometryChange()
        self._crop_item.img_w = w
        self._crop_item.img_h = h
        self._crop_item.crop_rect = QRectF(w * 0.1, h * 0.1, w * 0.8, h * 0.8)
        self._crop_item.update()

    def set_split_view_active(self, active: bool):
        self._item.split_active = active
        self._item.update()
        self.viewport().update()

    def set_split_orientation(self, orientation: str):
        self._item.split_orientation = orientation
        self._item.update()
        self.viewport().update()

    def set_split_inverted(self, inverted: bool):
        self._item.split_inverted = inverted
        self._item.update()
        self.viewport().update()

    def set_loupe_zoom(self, zoom: float):
        self._loupe_zoom = max(1.0, float(zoom))
        if self._loupe_mode or self._alt_pressed:
            self.viewport().update()

    def _any_tool_active(self) -> bool:
        return (self._loupe_mode or self._crop_mode or self._profile_mode
                or self._psf_mode or self._ruler_mode)

    def _update_interaction_state(self):
        """Единая точка выбора режима перетаскивания и курсора по набору активных инструментов."""
        if self._any_tool_active():
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            if self._crop_mode:
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().update()

    def reset_to_default_tool(self):
        """Полный сброс к инструменту по умолчанию (Pan): все режимы и временная лупа Alt выключены."""
        self._loupe_mode = False
        self._crop_mode = False
        self._profile_mode = False
        self._psf_mode = False
        self._ruler_mode = False
        self._ruler_dragging = False
        self._profile_dragging = False
        self._alt_pressed = False
        self._profile_p1 = None
        self._profile_p2 = None
        if self._crop_item is not None:
            self._crop_item.setVisible(False)
        self._update_interaction_state()

    def set_loupe_mode(self, active: bool):
        self._loupe_mode = active
        self._update_interaction_state()

    def set_crop_mode(self, active: bool):
        self._crop_mode = active
        if active:
            w = self._item.boundingRect().width()
            h = self._item.boundingRect().height()
            if self._crop_item is None:
                self._crop_item = CropOverlayItem(w, h)
                self._scene.addItem(self._crop_item)
            else:
                self._crop_item.img_w = w
                self._crop_item.img_h = h
                self._crop_item.crop_rect = QRectF(w * 0.1, h * 0.1, w * 0.8, h * 0.8)
                self._crop_item.setVisible(True)
        elif self._crop_item is not None:
            self._crop_item.setVisible(False)
        self._update_interaction_state()

    def set_profile_mode(self, active: bool):
        self._profile_mode = active
        if not active:
            self._profile_p1 = None
            self._profile_p2 = None
        self._update_interaction_state()

    def set_psf_mode(self, active: bool):
        self._psf_mode = active
        self._update_interaction_state()

    @property
    def zoom_factor(self) -> float:
        """Текущий коэффициент зума отображения кадра."""
        return max(1e-4, float(self.transform().m11()))

    @property
    def pan_offset(self) -> QPointF:
        """Текущее смещение панорамирования холста (QPointF)."""
        t = self.transform()
        return QPointF(t.dx(), t.dy())

    def apply_view_transform(self, painter: QPainter):
        """Применяет матрицу панорамирования и зума холста к QPainter."""
        painter.setTransform(self.transform())

    @property
    def image_viewport_rect(self) -> QRectF:
        """Экранные границы кадра с учетом отступов центрирования (letterbox) и зума."""
        if getattr(self, '_custom_image_viewport_rect', None) is not None:
            return self._custom_image_viewport_rect
        pix = None
        if hasattr(self, '_item') and self._item is not None:
            if not self._item.processed_pixmap.isNull():
                pix = self._item.processed_pixmap
            elif not self._item.raw_pixmap.isNull():
                pix = self._item.raw_pixmap
        if pix is not None and not pix.isNull() and pix.width() > 0 and pix.height() > 0:
            tl = self.mapFromScene(QPointF(0.0, 0.0))
            zf = self.zoom_factor
            return QRectF(tl.x(), tl.y(), pix.width() * zf, pix.height() * zf)
        vp = self.viewport()
        w = float(vp.width()) if vp else 800.0
        h = float(vp.height()) if vp else 600.0
        return QRectF(0.0, 0.0, w, h)

    @image_viewport_rect.setter
    def image_viewport_rect(self, val: Optional[QRectF]):
        self._custom_image_viewport_rect = val

    def screen_to_image(self, screen_pos: QPointF | QPoint) -> QPointF:
        """Переводит координаты экрана/вьюпорта в пиксели изображения с учетом letterbox."""
        rect = self.image_viewport_rect
        zf = max(1e-4, self.zoom_factor)
        pt = screen_pos.toPointF() if isinstance(screen_pos, QPoint) else screen_pos
        img_x = (pt.x() - rect.left()) / zf
        img_y = (pt.y() - rect.top()) / zf
        if math.isnan(img_x) or math.isnan(img_y) or math.isinf(img_x) or math.isinf(img_y):
            return QPointF(0.0, 0.0)
        return QPointF(img_x, img_y)

    def image_to_screen(self, img_pos: QPointF | QPoint) -> QPointF:
        """Переводит пиксели исходного изображения в координаты экрана/вьюпорта."""
        rect = self.image_viewport_rect
        zf = self.zoom_factor
        pt = img_pos.toPointF() if isinstance(img_pos, QPoint) else img_pos
        screen_x = rect.left() + pt.x() * zf
        screen_y = rect.top() + pt.y() * zf
        return QPointF(screen_x, screen_y)

    @property
    def disk_rect_img(self) -> Optional[QRectF]:
        """Координаты габаритной рамки диска Луны строго в пространстве изображения."""
        return self._disk_rect_img

    @disk_rect_img.setter
    def disk_rect_img(self, val: Optional[QRectF]):
        self._disk_rect_img = val

    @property
    def temp_disk_rect(self) -> Optional[QRectF]:
        """Псевдоним для disk_rect_img для обратной совместимости."""
        return self._disk_rect_img

    @temp_disk_rect.setter
    def temp_disk_rect(self, val: Optional[QRectF]):
        self._disk_rect_img = val

    @property
    def p1_img(self) -> Optional[QPointF]:
        """Точка 1 на дуге лимба в координатах изображения."""
        pts = self.disk_points if len(self.disk_points) > 0 else self._disk_points
        return pts[0] if len(pts) > 0 else None

    @property
    def p2_img(self) -> Optional[QPointF]:
        """Точка 2 на дуге лимба в координатах изображения."""
        pts = self.disk_points if len(self.disk_points) > 0 else self._disk_points
        return pts[1] if len(pts) > 1 else None

    @property
    def p3_img(self) -> Optional[QPointF]:
        """Точка 3 на дуге лимба в координатах изображения."""
        pts = self.disk_points if len(self.disk_points) > 0 else self._disk_points
        return pts[2] if len(pts) > 2 else None

    @property
    def _calibrated_disk(self):
        return self.calibrated_disk

    @_calibrated_disk.setter
    def _calibrated_disk(self, val):
        self.calibrated_disk = val

    @property
    def disk_mode(self) -> bool:
        return self._ruler_mode and (self._meas_submode == "disk" or getattr(self, 'active_mode', '') == "moon_disk")

    @disk_mode.setter
    def disk_mode(self, val: bool):
        if val:
            self.set_ruler_mode(True)
            self.set_measurement_submode("disk")
        else:
            if self._meas_submode == "disk":
                self.set_measurement_submode("ruler")

    def viewport_to_image_coords(self, pos: QPoint | QPointF) -> QPointF:
        """Безопасный перевод экранных координат вьюпорта в пиксели кадра."""
        return self.screen_to_image(pos)

    def set_ruler_mode(self, active: bool):
        self._ruler_mode = active
        if active:
            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                self.measurement_hud.show()
                self.measurement_hud.set_collapsed(False)
                self._update_measurement_hud_pos()
                if self._ruler_p1 is not None and self._ruler_p2 is not None and self._meas_submode == "ruler":
                    self._emit_ruler_info()
                elif self._ellipse_data is not None and self._meas_submode == "ellipse":
                    d = self._ellipse_data
                    self.measurement_hud.set_ellipse_results(
                        d["true_diameter_km"], d["apparent_diameter_km"], d["tilt_deg"],
                        d["cos_theta"], d["eccentricity"], d["semi_major_a"], d["semi_minor_b"]
                    )
        else:
            self._ruler_dragging = False
            self.is_drawing_disk = False
            self._disk_draft_dragging = False
            self._disk_transform_mode = None
            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                self.measurement_hud.hide()
        self._update_interaction_state()
        self.viewport().update()

    def set_measurement_submode(self, mode: str):
        if mode in ("moon_disk", "disk"):
            canonical = "disk"
            self.active_mode = "moon_disk"
        elif mode in ("crater_ellipse", "ellipse"):
            canonical = "ellipse"
            self.active_mode = "crater_ellipse"
        else:
            canonical = "ruler"
            self.active_mode = "ruler"

        self._meas_submode = canonical
        if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
            self.measurement_hud.set_mode(canonical)
        self.rulerSubmodeChanged.emit(canonical)
        if canonical == "ruler":
            self.set_status_text("Прямая линейка: зажмите ЛКМ и протяните отрезок (M)")
            if self._ruler_p1 is not None and self._ruler_p2 is not None:
                self._emit_ruler_info()
            elif hasattr(self, 'measurement_hud'):
                self.measurement_hud.clear_results()
        elif canonical == "disk":
            self.set_status_text("Привязка диска: растяните рамку 1:1 или укажите 3 точки на лимбе")
            if hasattr(self, 'measurement_hud'):
                self.measurement_hud.clear_results()
        elif canonical == "ellipse":
            self.set_status_text("Эллипс кратера: протяните максимальный поперечник (вал), затем растяните малую ось")
            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                self.measurement_hud.set_ellipse_step(self.ellipse_stage if self.ellipse_stage > 0 else 1)
            if self._ellipse_data is not None and hasattr(self, 'measurement_hud'):
                d = self._ellipse_data
                self.measurement_hud.set_ellipse_results(
                    d["true_diameter_km"], d["apparent_diameter_km"], d["tilt_deg"],
                    d["cos_theta"], d["eccentricity"], d["semi_major_a"], d["semi_minor_b"]
                )
            elif hasattr(self, 'measurement_hud'):
                self.measurement_hud.clear_results()
        self.viewport().update()

    def _hit_test_disk_rect(self, img_pos: QPointF) -> Optional[str]:
        """
        Проверяет попадание точки в координатах изображения в маркеры трансформации диска.
        Возвращает: 'tl', 'tr', 'br', 'bl', 'center' или None.
        """
        if img_pos is None:
            return None

        rect = self.temp_disk_rect
        if rect is None:
            c = self.disk_center if self.disk_center is not None else self._disk_draft_center
            r = self.disk_radius if self.disk_radius > 0.0 else self._disk_draft_radius
            if c is not None and r >= 5.0:
                rect = QRectF(c.x() - r, c.y() - r, 2.0 * r, 2.0 * r)
            else:
                return None

        if rect.width() < 5.0:
            return None

        zf = max(1e-4, getattr(self, 'zoom_factor', 1.0))
        tol = 12.0 / zf

        x = img_pos.x()
        y = img_pos.y()

        # 1. Проверка 4 угловых маркеров масштабирования (Resize Handles)
        if abs(x - rect.left()) <= tol and abs(y - rect.top()) <= tol:
            return "tl"
        if abs(x - rect.right()) <= tol and abs(y - rect.top()) <= tol:
            return "tr"
        if abs(x - rect.right()) <= tol and abs(y - rect.bottom()) <= tol:
            return "br"
        if abs(x - rect.left()) <= tol and abs(y - rect.bottom()) <= tol:
            return "bl"

        # 2. Проверка границы окружности диска (Resize по периметру)
        c = rect.center()
        dist_to_c = math.hypot(x - c.x(), y - c.y())
        radius = rect.width() / 2.0
        if abs(dist_to_c - radius) <= tol:
            return "edge"

        # 3. Проверка перемещения центра (Move / Pan Center)
        if dist_to_c < max(5.0, radius - tol):
            return "center"

        return None

    def _hit_test_disk_handles(self, vp_pos: QPoint) -> Optional[str]:
        """Возвращает имя маркера трансформации ('center', 'tl', 'tr', 'br', 'bl') или None."""
        img_pos = self.viewport_to_image_coords(vp_pos)
        return self._hit_test_disk_rect(img_pos)

    def _cursor_for_disk_handle(self, handle: Optional[str]) -> Qt.CursorShape:
        """Определяет форму курсора для Figma маркера трансформации."""
        if handle == "center":
            return Qt.CursorShape.SizeAllCursor
        elif handle in ("tl", "br"):
            return Qt.CursorShape.SizeFDiagCursor
        elif handle in ("tr", "bl"):
            return Qt.CursorShape.SizeBDiagCursor
        elif handle == "edge":
            return Qt.CursorShape.SizeAllCursor
        return Qt.CursorShape.CrossCursor

    def set_disk_method(self, method: str):
        if getattr(self, '_in_disk_method_update', False):
            return
        self._in_disk_method_update = True
        try:
            self._disk_method = method
            self.disk_points.clear()
            if self._disk_points is not self.disk_points:
                self._disk_points.clear()
            if hasattr(self, 'three_points') and self.three_points is not self.disk_points:
                self.three_points.clear()
            self.three_points_result = None
            self.disk_center = None
            self._disk_draft_center = None
            self.disk_radius = 0.0
            self._disk_draft_radius = 0.0
            self.temp_disk_rect = None
            self.pending_disk = None
            self._disk_state = "idle"
            self._disk_active_handle = None
            self._disk_hover_handle = None
            self.is_drawing_disk = False
            self._disk_draft_dragging = False
            self._disk_transform_mode = None
            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                self.measurement_hud.btn_apply_disk.setEnabled(False)
                if self.measurement_hud._disk_method != method:
                    self.measurement_hud.set_disk_method(method)
            if method == "three_points":
                self.set_status_text("Привязка диска: укажите 3 точки на дуге лимба Луны")
            else:
                self.set_status_text("Привязка диска: растяните рамку 1:1 по лимбу Луны")
            self.viewport().update()
        finally:
            self._in_disk_method_update = False

    def apply_disk_alignment(self):
        if self.pending_disk is not None:
            self.calibrated_disk = self.pending_disk
        elif self.temp_disk_rect and self.temp_disk_rect.width() > 5:
            c = self.temp_disk_rect.center()
            r = self.temp_disk_rect.width() / 2.0
            self.calibrated_disk = (c.x(), c.y(), r)
        elif self._disk_method == "center_radius":
            center = self.disk_center if self.disk_center is not None else self._disk_draft_center
            radius = self.disk_radius if self.disk_radius > 0.0 else self._disk_draft_radius
            if center is not None and radius >= 5.0:
                self.calibrated_disk = (center.x(), center.y(), radius)
        elif self._disk_method == "three_points":
            pts = self.disk_points if len(self.disk_points) == 3 else self._disk_points
            if len(pts) == 3:
                res = circle_from_3_points(pts[0], pts[1], pts[2])
                if res is not None:
                    c, r_disk = res
                    self.calibrated_disk = (c.x(), c.y(), r_disk)

        if self.calibrated_disk is not None:
            x0, y0, r_disk = self.calibrated_disk
            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                self.measurement_hud.set_disk_status(True, r_disk)
            self.set_status_text(f"Диск Луны откалиброван: R={int(round(r_disk))} px. Режим: Линейка.")
            self.temp_disk_rect = None
            self.pending_disk = None
            self.disk_is_dragging = False
            self.disk_drag_start = None
            self.disk_drag_current = None
            self._disk_state = "idle"
            self._disk_active_handle = None
            self._disk_hover_handle = None
            self._disk_draw_start = None
            self._disk_anchor = None
            self.disk_center = None
            self._disk_draft_center = None
            self.disk_radius = 0.0
            self._disk_draft_radius = 0.0
            self.disk_points.clear()
            self._disk_points.clear()
            # Автоматически переключаем инструмент обратно на линейку
            self.set_measurement_submode("ruler")
            self.viewport().update()

    def reset_disk_alignment(self):
        self.calibrated_disk = None
        self.pending_disk = None
        self.temp_disk_rect = None
        self.disk_is_dragging = False
        self.disk_drag_start = None
        self.disk_drag_current = None
        self._disk_state = "idle"
        self._disk_active_handle = None
        self._disk_hover_handle = None
        self._disk_draw_start = None
        self._disk_anchor = None
        self.disk_center = None
        self._disk_draft_center = None
        self.disk_radius = 0.0
        self._disk_draft_radius = 0.0
        self.disk_points.clear()
        if self._disk_points is not self.disk_points:
            self._disk_points.clear()
        if hasattr(self, 'three_points') and self.three_points is not self.disk_points:
            self.three_points.clear()
        self.three_points_result = None
        if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
            self.measurement_hud.set_disk_status(False)
            self.measurement_hud.set_disk_prompt("", can_apply=False)
            self.measurement_hud.clear_results()
        self.set_status_text("Калибровка диска Луны сброшена")
        if self._ruler_p1 is not None and self._ruler_p2 is not None:
            self._emit_ruler_info()
        self.viewport().update()

    def set_telescope_params(self, focal_len_mm: float, pixel_size_um: float):
        self._telescope_focal_len = max(1.0, float(focal_len_mm))
        self._telescope_pixel_size = max(0.1, float(pixel_size_um))
        if self._ruler_p1 is not None and self._ruler_p2 is not None:
            self._emit_ruler_info()
        self.viewport().update()

    def set_lunar_distance(self, distance_km: float):
        self._lunar_distance_km = max(100000.0, float(distance_km))
        if self._ruler_p1 is not None and self._ruler_p2 is not None:
            self._emit_ruler_info()
        self.viewport().update()

    def set_ruler_unit(self, unit: str):
        self._ruler_unit = str(unit).lower()
        if self._ruler_p1 is not None and self._ruler_p2 is not None:
            self._emit_ruler_info()
        self.viewport().update()

    def clear_ruler(self):
        self._ruler_p1 = None
        self._ruler_p2 = None
        self._ruler_dragging = False
        self.ellipse_stage = 0
        self._ellipse_step = 0
        self.ellipse_start = None
        self._ellipse_p_start = None
        self.ellipse_end = None
        self._ellipse_p_end = None
        self._ellipse_cursor = None
        self._ellipse_data = None
        self.ellipse_center = None
        self.ellipse_a = 0.0
        self.ellipse_b = 0.0
        self.ellipse_angle = 0.0
        if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
            self.measurement_hud.clear_results()
        self.viewport().update()

    def clear_psf_marker(self):
        self._psf_last_pos = None
        self.viewport().update()

    def _format_ruler_dist(self, dist_px: float) -> tuple[str, str]:
        scale = 206.265 * self._telescope_pixel_size / self._telescope_focal_len
        ang_arcsec = dist_px * scale
        if ang_arcsec < 60.0:
            ang_str = f"{ang_arcsec:.1f}\""
        elif ang_arcsec < 3600.0:
            ang_str = f"{ang_arcsec / 60.0:.2f}'"
        else:
            ang_str = f"{ang_arcsec / 3600.0:.2f}°"

        lunar_dist = getattr(self, '_lunar_distance_km', 384400.0)
        # Distance_km = (Pixels * (Pixel_Size_um / (Focal_Length_mm * 1000))) * Current_Moon_Distance_km
        km = (dist_px * (self._telescope_pixel_size / (self._telescope_focal_len * 1000.0))) * lunar_dist

        unit = getattr(self, '_ruler_unit', 'all')
        if unit == 'px':
            dist_str = f"{dist_px:.1f} px"
        elif unit in ('arcsec', 'arc'):
            dist_str = f"{ang_str} ({ang_arcsec:.2f}\")"
        elif unit == 'km':
            dist_str = f"{km:.1f} km ({ang_str})"
        else:
            dist_str = f"{dist_px:.1f} px ({ang_str})"
        return dist_str, ang_str

    def _emit_ruler_info(self):
        if self._ruler_p1 is None or self._ruler_p2 is None:
            return
        dx = self._ruler_p2.x() - self._ruler_p1.x()
        dy = self._ruler_p2.y() - self._ruler_p1.y()
        dist_px = float(np.hypot(dx, dy))
        angle_deg = float(np.degrees(np.arctan2(-dy, dx)))
        if angle_deg < 0:
            angle_deg += 360.0
        if abs(angle_deg) < 1e-4 or abs(angle_deg - 360.0) < 1e-4:
            angle_deg = 0.0

        dist_str, ang_str = self._format_ruler_dist(dist_px)

        lunar_dist = getattr(self, '_lunar_distance_km', 384400.0)
        scale_km_per_px = (self._telescope_pixel_size / (self._telescope_focal_len * 1000.0)) * lunar_dist
        proj_km = dist_px * scale_km_per_px

        true_km = None
        tilt_deg = 0.0
        cos_theta = 1.0

        if self._calibrated_disk is not None:
            x0, y0, r_disk = self._calibrated_disk
            corr = calculate_spherical_correction(
                (self._ruler_p1.x(), self._ruler_p1.y()),
                (self._ruler_p2.x(), self._ruler_p2.y()),
                (x0, y0),
                r_disk,
                scale_km_per_px
            )
            true_km = corr["true_km"]
            tilt_deg = corr["tilt_deg"]
            cos_theta = corr["cos_theta"]
            if true_km is not None:
                summary = f"Proj: {dist_str} | True: {true_km:.1f} km (θ={tilt_deg:.0f}°, cos θ={cos_theta:.2f}) | {angle_deg:.1f}°"
            else:
                summary = f"Proj: {dist_str} | True: N/A (Limb boundary) | {angle_deg:.1f}°"
            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                self.measurement_hud.set_ruler_results(
                    proj_km, ang_str, dist_px, true_km, tilt_deg, cos_theta, is_calibrated=True
                )
        else:
            summary = f"L: {dist_str} | Angle: {angle_deg:.1f}°"
            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                self.measurement_hud.set_ruler_results(
                    proj_km, ang_str, dist_px, is_calibrated=False
                )

        self.rulerMeasured.emit(dist_px, angle_deg, summary)
        self.set_status_text(f"Ruler: {summary}")

    def _update_ellipse_calc(self, cursor_pos: QPointF):
        p_start = self.ellipse_start if self.ellipse_start is not None else self._ellipse_p_start
        p_end = self.ellipse_end if self.ellipse_end is not None else self._ellipse_p_end
        if p_start is None or p_end is None:
            return
        dx = p_end.x() - p_start.x()
        dy = p_end.y() - p_start.y()
        dist_2a = math.hypot(dx, dy)
        if dist_2a <= 1e-4:
            return

        p1 = (p_start.x(), p_start.y())
        p2 = (p_end.x(), p_end.y())
        p_cur = (cursor_pos.x(), cursor_pos.y())
        lunar_dist = getattr(self, '_lunar_distance_km', 384400.0)
        scale_km_per_px = (self._telescope_pixel_size / (self._telescope_focal_len * 1000.0)) * lunar_dist
        try:
            self._ellipse_data = calculate_crater_ellipse(
                p1, p2, p_cur, scale_km_per_px
            )
        except Exception:
            traceback.print_exc()
            return

        if self._ellipse_data and hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
            d = self._ellipse_data
            self.ellipse_center = QPointF(d["center"][0], d["center"][1])
            self.ellipse_a = d["semi_major_a"]
            self.ellipse_b = d["semi_minor_b"]
            self.ellipse_angle = d["angle_deg"]
            self.measurement_hud.set_ellipse_results(
                d["true_diameter_km"], d["apparent_diameter_km"], d["tilt_deg"],
                d["cos_theta"], d["eccentricity"], d["semi_major_a"], d["semi_minor_b"]
            )

    def apply_current_crop(self):
        if self._crop_mode and self._crop_item is not None:
            r = self._crop_item.crop_rect.normalized()
            self.cropApplied.emit(r)

    def paintEvent(self, event):
        """Отрисовка вьюпорта с аппаратной синхронизацией Split View."""
        super().paintEvent(event)

    def drawForeground(self, painter: QPainter, rect: QRectF):
        super().drawForeground(painter, rect)

        if self._item.processed_pixmap.isNull() and self._item.raw_pixmap.isNull():
            return
        pix = self._item.processed_pixmap if not self._item.processed_pixmap.isNull() else self._item.raw_pixmap
        w_scene = pix.width()
        h_scene = pix.height()
        if w_scene <= 0 or h_scene <= 0:
            return

        top_left_vp = self.mapFromScene(QPointF(0, 0))
        bottom_right_vp = self.mapFromScene(QPointF(w_scene, h_scene))

        img_left = min(top_left_vp.x(), bottom_right_vp.x())
        img_right = max(top_left_vp.x(), bottom_right_vp.x())
        img_top = min(top_left_vp.y(), bottom_right_vp.y())
        img_bottom = max(top_left_vp.y(), bottom_right_vp.y())

        # Статическая векторная маска клиппинга (в координатах сцены / пикселей снимка)
        if self.clipping_overlay_mode is not None and self._clipping_mask_pixmap is not None:
            painter.drawPixmap(0, 0, self._clipping_mask_pixmap)

        painter.save()
        painter.resetTransform()

        vp_w = self.viewport().width()
        vp_h = self.viewport().height()

        # 1. Линия Split View и плашки Before / After
        if self._item.split_active and not self._item.processed_pixmap.isNull():
            accent_col = QColor(self._theme.get('bright', '#00fff2'))
            pen = QPen(accent_col, 2)
            pen.setCosmetic(True)
            painter.setPen(pen)

            lbl_first = "AFTER" if self._item.split_inverted else "BEFORE"
            lbl_second = "BEFORE" if self._item.split_inverted else "AFTER"

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            fm = painter.fontMetrics()

            def draw_badge(text: str, x: float, y: float, is_after: bool):
                text_w = fm.horizontalAdvance(text)
                text_h = fm.height()
                pad_x, pad_y = 8, 3
                bg_rect = QRectF(x, y, text_w + pad_x * 2, text_h + pad_y * 2)

                border_col = accent_col if is_after else QColor("#555566")
                painter.setPen(QPen(border_col, 1))
                painter.setBrush(QBrush(QColor(18, 18, 24, 230)))
                painter.drawRoundedRect(bg_rect, 5, 5)

                text_col = accent_col if is_after else QColor("#d0d0dc")
                painter.setPen(text_col)
                painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, text)

            b1_w = fm.horizontalAdvance(lbl_first) + 16
            b2_w = fm.horizontalAdvance(lbl_second) + 16
            b_h = fm.height() + 6

            if self._item.split_orientation == "Vertical":
                scene_x = w_scene * self._item.split_fraction
                view_x = self.mapFromScene(QPointF(scene_x, 0)).x()
                split_pos_x = int(round(view_x))

                line_y1 = max(0.0, img_top)
                line_y2 = min(float(vp_h), img_bottom)
                if line_y2 > line_y1:
                    painter.drawLine(split_pos_x, int(line_y1), split_pos_x, int(line_y2))
                    # Терминатор-сплиттер: серп смотрит в сторону активного слоя "After"
                    angle = 0.0 if lbl_second == "AFTER" else 180.0
                    self._draw_lunar_phase_handle(painter, QPointF(split_pos_x, (line_y1 + line_y2) / 2.0), angle, accent_col)

                badge_y = max(img_top, 0.0) + 10
                badge_y = min(badge_y, max(0.0, img_bottom - b_h - 10))

                x_left = split_pos_x - b1_w - 12
                x_right = split_pos_x + 12
                x_left = max(8, min(x_left, vp_w - b1_w - 8))
                x_right = max(8, min(x_right, vp_w - b2_w - 8))

                draw_badge(lbl_first, x_left, badge_y, lbl_first == "AFTER")
                draw_badge(lbl_second, x_right, badge_y, lbl_second == "AFTER")
            else:
                scene_y = h_scene * self._item.split_fraction
                view_y = self.mapFromScene(QPointF(0, scene_y)).y()
                split_pos_y = int(round(view_y))

                line_x1 = max(0.0, img_left)
                line_x2 = min(float(vp_w), img_right)
                if line_x2 > line_x1:
                    painter.drawLine(int(line_x1), split_pos_y, int(line_x2), split_pos_y)
                    # При горизонтальной линии серп смотрит вниз (90°) или вверх (270°) к слою "After"
                    angle = 90.0 if lbl_second == "AFTER" else 270.0
                    self._draw_lunar_phase_handle(painter, QPointF((line_x1 + line_x2) / 2.0, split_pos_y), angle, accent_col)

                badge_x = max(img_left, 0.0) + 12
                badge_x = min(badge_x, max(0.0, img_right - 80))

                y_top = split_pos_y - b_h - 10
                y_bottom = split_pos_y + 10
                y_top = max(8, min(y_top, vp_h - b_h - 8))
                y_bottom = max(8, min(y_bottom, vp_h - b_h - 8))

                draw_badge(lbl_first, badge_x, y_top, lbl_first == "AFTER")
                draw_badge(lbl_second, badge_x, y_bottom, lbl_second == "AFTER")

        # 2. Отрезок Profile Line Tool
        if self._profile_p1 is not None and self._profile_p2 is not None:
            vp_p1 = self.mapFromScene(self._profile_p1)
            vp_p2 = self.mapFromScene(self._profile_p2)
            prof_pen = QPen(QColor("#ff007f"), 2, Qt.PenStyle.SolidLine)
            prof_pen.setCosmetic(True)
            painter.setPen(prof_pen)
            painter.drawLine(vp_p1, vp_p2)

            painter.setBrush(QBrush(QColor("#ff007f")))
            painter.drawEllipse(vp_p1, 4, 4)
            painter.drawEllipse(vp_p2, 4, 4)

        # 2.1. Мишень инструмента PSF / FWHM
        if self._psf_mode and self._psf_last_pos is not None:
            vp_psf = self.mapFromScene(self._psf_last_pos)
            cx, cy = vp_psf.x(), vp_psf.y()
            r_ret = 9.0
            painter.setPen(QPen(QColor("#00fff2"), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r_ret, r_ret)
            painter.drawLine(QPointF(cx - r_ret - 4, cy), QPointF(cx - r_ret + 1, cy))
            painter.drawLine(QPointF(cx + r_ret - 1, cy), QPointF(cx + r_ret + 4, cy))
            painter.drawLine(QPointF(cx, cy - r_ret - 4), QPointF(cx, cy - r_ret + 1))
            painter.drawLine(QPointF(cx, cy + r_ret - 1), QPointF(cx, cy + r_ret + 4))

        # 2.2.0 Отрисовка интерактивного диска Луны (Bounding Box) или откалиброванного ориентира
        is_disk_submode = (self._meas_submode == "disk" or getattr(self, 'active_mode', '') == "moon_disk")
        is_ruler_submode = (self._meas_submode == "ruler" or getattr(self, 'active_mode', '') == "ruler")
        is_ellipse_submode = (self._meas_submode == "ellipse" or getattr(self, 'active_mode', '') == "crater_ellipse")

        if self._ruler_mode and (is_disk_submode or (is_ruler_submode and self.calibrated_disk)):
            try:
                draw_rect = self.disk_rect_img or self.temp_disk_rect
                if draw_rect is None and is_disk_submode and self.disk_center is not None and self.disk_radius > 5:
                    r = self.disk_radius
                    draw_rect = QRectF(self.disk_center.x() - r, self.disk_center.y() - r, 2.0 * r, 2.0 * r)
                    self.disk_rect_img = draw_rect

                if is_disk_submode and draw_rect and draw_rect.width() > 5:
                    top_left_s = self.image_to_screen(draw_rect.topLeft())
                    bottom_right_s = self.image_to_screen(draw_rect.bottomRight())
                    screen_rect = QRectF(top_left_s, bottom_right_s).normalized()
                    center_s = screen_rect.center()

                    # Тонкая пунктирная рамка квадрата:
                    pen_box = QPen(QColor(100, 120, 150, 140), 1.0, Qt.PenStyle.DashLine)
                    painter.setPen(pen_box)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(screen_rect)

                    # 1. Окружность диска:
                    pen_circle = QPen(QColor(0, 229, 255, 200), 2.0)
                    painter.setPen(pen_circle)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(screen_rect)

                    # 2. Крестик центра (+):
                    cr = 6.0
                    painter.drawLine(QPointF(center_s.x() - cr, center_s.y()), QPointF(center_s.x() + cr, center_s.y()))
                    painter.drawLine(QPointF(center_s.x(), center_s.y() - cr), QPointF(center_s.x(), center_s.y() + cr))

                    # 3. 4 угловых маркера ресайза (белые квадратики с черной окантовкой):
                    h_size = 8.0
                    painter.setBrush(QBrush(QColor(255, 255, 255)))
                    painter.setPen(QPen(QColor(0, 0, 0), 1.0))
                    for pt in [screen_rect.topLeft(), screen_rect.topRight(),
                               screen_rect.bottomLeft(), screen_rect.bottomRight()]:
                        painter.drawRect(QRectF(pt.x() - h_size / 2.0, pt.y() - h_size / 2.0, h_size, h_size))

                # Отрисовка уже откалиброванного статического диска (в режиме ruler тонкая линия 20% alpha):
                elif is_ruler_submode and self.calibrated_disk:
                    cx, cy, r = self.calibrated_disk
                    center_s = self.image_to_screen(QPointF(cx, cy))
                    r_s = r * self.zoom_factor
                    pen_calib = QPen(QColor(0, 229, 255, 50), 1.5, Qt.PenStyle.DotLine)
                    painter.setPen(pen_calib)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(center_s, r_s, r_s)
            except Exception:
                traceback.print_exc()

        if self._ruler_mode and is_disk_submode and self._disk_method == "three_points" and self.temp_disk_rect is None:
            try:
                pts = self.disk_points if len(self.disk_points) > 0 else self._disk_points
                pt_pen = QPen(QColor("#00e5ff"), 2.0)
                painter.setPen(pt_pen)
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                for i, pt in enumerate(pts):
                    s_pt = self.image_to_screen(pt)
                    painter.setBrush(QBrush(QColor("#00adb5")))
                    painter.drawEllipse(s_pt, 5.0, 5.0)
                    painter.setPen(QColor("#ffffff"))
                    painter.drawText(QRectF(s_pt.x() + 7.0, s_pt.y() - 10.0, 30.0, 18.0), Qt.AlignmentFlag.AlignLeft, f"P{i+1}")
                    painter.setPen(pt_pen)

                # Превью окружности по 2 точкам + курсор или 3 точкам
                pts_to_fit = list(pts)
                if len(pts_to_fit) == 2 and hasattr(self, '_mouse_vp') and self._mouse_vp is not None:
                    scene_cur = self.viewport_to_image_coords(self._mouse_vp.toPoint())
                    d1 = float(np.hypot(scene_cur.x() - pts_to_fit[0].x(), scene_cur.y() - pts_to_fit[0].y()))
                    d2 = float(np.hypot(scene_cur.x() - pts_to_fit[1].x(), scene_cur.y() - pts_to_fit[1].y()))
                    if d1 > 15.0 and d2 > 15.0:
                        pts_to_fit.append(scene_cur)

                if len(pts_to_fit) == 3:
                    c_fit = circle_from_3_points(pts_to_fit[0], pts_to_fit[1], pts_to_fit[2])
                    if c_fit is not None:
                        center, r = c_fit
                        if r > 1.0:
                            center_s = self.image_to_screen(center)
                            r_s = r * self.zoom_factor
                            style = Qt.PenStyle.DashLine if len(pts) < 3 else Qt.PenStyle.SolidLine
                            c_pen = QPen(QColor("#00e5ff"), 1.8, style)
                            painter.setPen(c_pen)
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                            painter.drawEllipse(center_s, r_s, r_s)
            except Exception:
                traceback.print_exc()

        # 2.2.2 Замер кратера эллиптическим инструментом (Crater Ellipse)
        p_start = self.ellipse_start if self.ellipse_start is not None else self._ellipse_p_start
        p_end = self.ellipse_end if self.ellipse_end is not None else self._ellipse_p_end
        stage = self.ellipse_stage if self.ellipse_stage > 0 else self._ellipse_step

        if self._ruler_mode and is_ellipse_submode and p_start is not None:
            try:
                painter.save()
                view_scale = max(1e-4, self.transform().m11())
                p1_vp = self.mapFromScene(p_start)

                if stage == 1 and p_end is not None:
                    p2_vp = self.mapFromScene(p_end)
                    if not (math.isnan(p1_vp.x()) or math.isnan(p1_vp.y()) or math.isnan(p2_vp.x()) or math.isnan(p2_vp.y())):
                        axis_pen = QPen(QColor("#00e5ff"), 1.8)
                        axis_pen.setCosmetic(True)
                        painter.setPen(axis_pen)
                        painter.drawLine(p1_vp, p2_vp)
                        painter.setBrush(QBrush(QColor("#00e5ff")))
                        painter.drawEllipse(p1_vp, 3.5, 3.5)
                        painter.drawEllipse(p2_vp, 3.5, 3.5)

                elif stage in (2, 3) and self._ellipse_data is not None:
                    d = self._ellipse_data
                    xc, yc = d["center"]
                    a = self.ellipse_a if self.ellipse_a > 0.0 else d["semi_major_a"]
                    b = self.ellipse_b if self.ellipse_b > 0.0 else d["semi_minor_b"]
                    angle_deg = self.ellipse_angle if self.ellipse_angle != 0.0 else d["angle_deg"]
                    true_km = d["true_diameter_km"]
                    apparent_km = d["apparent_diameter_km"]
                    tilt_deg = d["tilt_deg"]
                    cos_theta = d["cos_theta"]

                    # Рисовать эллипс только если a > 1.0 and b > 0.5; исключить передачу NaN/Inf в rotate и drawEllipse
                    if (a > 1.0 and b > 0.5 and
                        not (math.isnan(xc) or math.isnan(yc) or math.isnan(a) or math.isnan(b) or
                             math.isnan(angle_deg) or math.isinf(a) or math.isinf(b) or math.isinf(angle_deg))):

                        vc = self.mapFromScene(QPointF(xc, yc))
                        a_vp = a * view_scale
                        b_vp = b * view_scale

                        if not (math.isnan(vc.x()) or math.isnan(vc.y()) or math.isinf(vc.x()) or math.isinf(vc.y())):
                            painter.save()
                            painter.translate(vc.x(), vc.y())
                            painter.rotate(angle_deg)

                            # Полупрозрачная заливка кратера
                            painter.setBrush(QBrush(QColor(0, 229, 255, 35)))
                            ell_pen = QPen(QColor("#00e5ff"), 2.0)
                            ell_pen.setCosmetic(True)
                            painter.setPen(ell_pen)
                            painter.drawEllipse(QPointF(0, 0), a_vp, b_vp)

                            # Главная ось со засечками
                            major_pen = QPen(QColor("#00e5ff"), 1.6)
                            major_pen.setCosmetic(True)
                            painter.setPen(major_pen)
                            painter.drawLine(QPointF(-a_vp, 0), QPointF(a_vp, 0))
                            painter.drawLine(QPointF(-a_vp, -5), QPointF(-a_vp, 5))
                            painter.drawLine(QPointF(a_vp, -5), QPointF(a_vp, 5))

                            # Малая ось (пунктир)
                            minor_pen = QPen(QColor("#38bdf8"), 1.2, Qt.PenStyle.DashLine)
                            minor_pen.setCosmetic(True)
                            painter.setPen(minor_pen)
                            painter.drawLine(QPointF(0, -b_vp), QPointF(0, b_vp))
                            painter.restore()

                            # Плашка результатов рядом с эллипсом с защитой от вылета за границы экрана
                            badge_text = f"True Dia: {true_km:.1f} km | Apparent: {apparent_km:.1f} km | Tilt: {tilt_deg:.0f}° (cos θ = {cos_theta:.2f})"
                            badge_x = vc.x() + a_vp + 12.0
                            badge_y = vc.y() - b_vp - 10.0

                            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                            fm = painter.fontMetrics()
                            bw = fm.horizontalAdvance(badge_text) + 12
                            bh = fm.height() + 6

                            # Viewport boundary clamping
                            badge_x = max(10.0, min(badge_x, float(vp_w) - bw - 10.0))
                            badge_y = max(10.0, min(badge_y, float(vp_h) - bh - 10.0))
                            b_rect = QRectF(badge_x, badge_y, bw, bh)

                            painter.setPen(QPen(QColor("#00e5ff"), 1))
                            painter.setBrush(QBrush(QColor(18, 18, 26, 230)))
                            painter.drawRoundedRect(b_rect, 5, 5)
                            painter.setPen(QColor("#e2e8f0"))
                            painter.drawText(b_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

                painter.restore()
            except Exception:
                traceback.print_exc()

        # 2.2.3 Линейка расстояний и углов (Pixel / Angular / Spherical Ruler)
        if (self._ruler_mode or self._ruler_dragging) and is_ruler_submode and self._ruler_p1 is not None and self._ruler_p2 is not None:
            vp1 = QPointF(self.mapFromScene(self._ruler_p1))
            vp2 = QPointF(self.mapFromScene(self._ruler_p2))
            dx = self._ruler_p2.x() - self._ruler_p1.x()
            dy = self._ruler_p2.y() - self._ruler_p1.y()
            dist_px = float(np.hypot(dx, dy))
            angle_deg = float(np.degrees(np.arctan2(-dy, dx)))
            if angle_deg < 0:
                angle_deg += 360.0
            if abs(angle_deg) < 1e-4 or abs(angle_deg - 360.0) < 1e-4:
                angle_deg = 0.0

            ruler_pen = QPen(QColor("#e5a93b"), 1.8)
            ruler_pen.setCosmetic(True)
            painter.setPen(ruler_pen)
            painter.drawLine(vp1, vp2)

            vx = vp2.x() - vp1.x()
            vy = vp2.y() - vp1.y()
            v_len = np.hypot(vx, vy)
            if v_len > 1.0:
                nx = -vy / v_len
                ny = vx / v_len
                t_size = 6.0
                painter.drawLine(QPointF(vp1.x() - nx * t_size, vp1.y() - ny * t_size),
                                 QPointF(vp1.x() + nx * t_size, vp1.y() + ny * t_size))
                painter.drawLine(QPointF(vp2.x() - nx * t_size, vp2.y() - ny * t_size),
                                 QPointF(vp2.x() + nx * t_size, vp2.y() + ny * t_size))

            painter.setBrush(QBrush(QColor("#e5a93b")))
            painter.drawEllipse(vp1, 2.5, 2.5)
            painter.drawEllipse(vp2, 2.5, 2.5)

            dist_str, _ = self._format_ruler_dist(dist_px)

            if self._calibrated_disk is not None:
                x0, y0, r_disk = self._calibrated_disk
                lunar_dist = getattr(self, '_lunar_distance_km', 384400.0)
                scale_km_per_px = (self._telescope_pixel_size / (self._telescope_focal_len * 1000.0)) * lunar_dist
                corr = calculate_spherical_correction(
                    (self._ruler_p1.x(), self._ruler_p1.y()),
                    (self._ruler_p2.x(), self._ruler_p2.y()),
                    (x0, y0),
                    r_disk,
                    scale_km_per_px
                )
                if corr["true_km"] is not None:
                    badge_text = f"Proj: {dist_str} | True: {corr['true_km']:.1f} km (θ={corr['tilt_deg']:.0f}°)"
                else:
                    badge_text = f"Proj: {dist_str} | True: N/A (Limb boundary)"
            else:
                badge_text = f"L: {dist_str} | {angle_deg:.1f}°"

            mid_x = (vp1.x() + vp2.x()) / 2.0
            mid_y = (vp1.y() + vp2.y()) / 2.0
            if v_len > 1.0:
                badge_x = mid_x + nx * 14.0
                badge_y = mid_y + ny * 14.0
            else:
                badge_x = mid_x + 10.0
                badge_y = mid_y - 10.0

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            fm = painter.fontMetrics()
            bw = fm.horizontalAdvance(badge_text) + 12
            bh = fm.height() + 6
            bx = badge_x - bw / 2.0
            by = badge_y - bh / 2.0

            # Viewport boundary clamping
            bx = max(10.0, min(bx, float(vp_w) - bw - 10.0))
            by = max(10.0, min(by, float(vp_h) - bh - 10.0))
            b_rect = QRectF(bx, by, bw, bh)

            painter.setPen(QPen(QColor("#e5a93b"), 1))
            painter.setBrush(QBrush(QColor(18, 18, 26, 225)))
            painter.drawRoundedRect(b_rect, 5, 5)

            painter.setPen(QColor("#ffe8a0"))
            painter.drawText(b_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        # 3. Интерактивная круглая лупа (инструмент Loupe или Alt)
        if (self._loupe_mode or self._alt_pressed) and not self._item.processed_pixmap.isNull():
            cx = self._mouse_vp.x()
            cy = self._mouse_vp.y()
            radius = 85.0
            zoom = getattr(self, '_loupe_zoom', 2.0)

            canvas_scale = max(1e-4, self.transform().m11())
            scene_pt = self.mapToScene(self._mouse_vp.toPoint())
            src_w = (2.0 * radius) / (canvas_scale * zoom)
            src_h = (2.0 * radius) / (canvas_scale * zoom)
            src_rect = QRectF(scene_pt.x() - src_w / 2.0, scene_pt.y() - src_h / 2.0, src_w, src_h)

            painter.save()
            clip_path = QPainterPath()
            clip_path.addEllipse(QPointF(cx, cy), radius, radius)
            painter.setClipPath(clip_path)

            painter.drawPixmap(
                QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius),
                self._item.processed_pixmap,
                src_rect
            )

            painter.setPen(QPen(QColor(0, 255, 242, 120), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(cx - radius), int(cy), int(cx + radius), int(cy))
            painter.drawLine(int(cx), int(cy - radius), int(cx), int(cy + radius))
            painter.drawEllipse(QPointF(cx, cy), 3, 3)
            painter.restore()

            painter.setPen(QPen(QColor("#00fff2"), 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#00fff2"))
            loupe_text = f"{int(zoom)}x LOUPE" if float(zoom).is_integer() else f"{zoom:.1f}x LOUPE"
            tw = painter.fontMetrics().horizontalAdvance(loupe_text)
            painter.drawText(int(cx - tw / 2.0), int(cy + radius - 8), loupe_text)

        # 4. Пульсирующая радарная анимация зон клиппинга (Sonar Ping Animation)
        if self.clipping_overlay_mode is not None and self._clipping_centroids:
            now = time.monotonic()
            elapsed = now - getattr(self, '_radar_start_time', now)
            cycle_duration = 1.2
            phase1 = (elapsed % cycle_duration) / cycle_duration
            phase2 = ((elapsed - 0.4) % cycle_duration) / cycle_duration

            base_hex = "#00b0ff" if self.clipping_overlay_mode == "shadows" else "#ff1744"
            base_color = QColor(base_hex)

            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            for scene_pt in self._clipping_centroids:
                vp_pt = self.mapFromScene(scene_pt)
                pt_f = QPointF(float(vp_pt.x()), float(vp_pt.y()))
                if pt_f.x() < -60 or pt_f.x() > vp_w + 60 or pt_f.y() < -60 or pt_f.y() > vp_h + 60:
                    continue

                for phase in (phase1, phase2):
                    r = 5.0 + 35.0 * phase
                    alpha = int(round(240.0 * (1.0 - phase)))
                    if alpha <= 0:
                        continue
                    ring_col = QColor(base_color)
                    ring_col.setAlpha(max(0, min(240, alpha)))
                    painter.setPen(QPen(ring_col, 2.0))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(pt_f, r, r)

            painter.restore()

        painter.restore()

    def _draw_lunar_phase_handle(self, painter: QPainter, center: QPointF, angle_deg: float, accent_col: QColor):
        """Отрисовывает миниатюрную стилизованную Луну с прорисованным серпом/терминатором."""
        R = 8.5
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.translate(center.x(), center.y())
        painter.rotate(angle_deg)

        # 1. Тонкое неоновое свечение активного стейта
        glow = QRadialGradient(0.0, 0.0, R + 4.0)
        glow.setColorAt(0.0, QColor(accent_col.red(), accent_col.green(), accent_col.blue(), 90))
        glow.setColorAt(0.7, QColor(accent_col.red(), accent_col.green(), accent_col.blue(), 25))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(QRectF(-R - 4.0, -R - 4.0, (R + 4.0) * 2.0, (R + 4.0) * 2.0), QBrush(glow))

        # 2. Тёмная сторона Луны: глубокий матовый графит (#1E2538) с тонким ободком
        painter.setPen(QPen(QColor(accent_col.red(), accent_col.green(), accent_col.blue(), 130), 1.0))
        painter.setBrush(QBrush(QColor("#1E2538")))
        painter.drawEllipse(QRectF(-R, -R, R * 2.0, R * 2.0))

        # 3. Освещённая сторона (серп/полумесяц) с плавной эллиптической линией терминатора
        crescent_path = QPainterPath()
        crescent_path.moveTo(0.0, -R)
        # Внешняя дуга: 12 часов (90°) -> 6 часов (270°) по часовой стрелке (-180°)
        crescent_path.arcTo(QRectF(-R, -R, R * 2.0, R * 2.0), 90.0, -180.0)
        # Внутренняя дуга терминатора: 6 часов (270°) -> 12 часов (90°) против часовой стрелки (180°)
        r_term = 0.35 * R
        crescent_path.arcTo(QRectF(-r_term, -R, r_term * 2.0, R * 2.0), 270.0, 180.0)
        crescent_path.closeSubpath()

        # Светлый лунный оттенок (#E2E8F0) с лёгким неоновым бирюзовым свечением
        lunar_grad = QLinearGradient(0.0, -R, R, R)
        lunar_grad.setColorAt(0.0, QColor("#FFFFFF"))
        lunar_grad.setColorAt(0.5, QColor("#E2E8F0"))
        lunar_grad.setColorAt(1.0, QColor(accent_col.red(), accent_col.green(), accent_col.blue(), 220))

        painter.fillPath(crescent_path, QBrush(lunar_grad))
        painter.strokePath(crescent_path, QPen(QColor(accent_col.red(), accent_col.green(), accent_col.blue(), 180), 0.75))

        painter.restore()

    def _is_near_split_line(self, scene_pos: QPointF) -> bool:
        if not self._item.split_active or self._item.processed_pixmap.isNull():
            return False

        if self._item.split_orientation == "Vertical":
            curr_x = self._item.processed_pixmap.width() * self._item.split_fraction
            view_curr = self.mapFromScene(QPointF(curr_x, 0)).x()
            view_mouse = self.mapFromScene(scene_pos).x()
            return abs(view_curr - view_mouse) <= 10
        else:
            curr_y = self._item.processed_pixmap.height() * self._item.split_fraction
            view_curr = self.mapFromScene(QPointF(0, curr_y)).y()
            view_mouse = self.mapFromScene(scene_pos).y()
            return abs(view_curr - view_mouse) <= 10

    def _sync_alt_state(self, modifiers):
        """Синхронизирует временную лупу Alt с реальным состоянием модификаторов (защита от залипания)."""
        alt_now = bool(modifiers & Qt.KeyboardModifier.AltModifier)
        if alt_now != self._alt_pressed:
            self._alt_pressed = alt_now
            self.viewport().update()

    def focusOutEvent(self, event):
        # Потеря фокуса (например, переход в окно инспектора) — KeyRelease Alt холст уже не получит
        if self._alt_pressed:
            self._alt_pressed = False
            self.viewport().update()
        super().focusOutEvent(event)

    def mousePressEvent(self, event):
        self._sync_alt_state(event.modifiers())
        pos = event.pos()
        img_pos = self.viewport_to_image_coords(pos)
        scene_pos = img_pos if img_pos is not None else self.mapToScene(pos)

        if self._loupe_mode and event.button() == Qt.MouseButton.LeftButton:
            self.viewport().update()
            event.accept()
            return

        if self._psf_mode and event.button() == Qt.MouseButton.LeftButton:
            self._psf_last_pos = scene_pos
            self.psfPointClicked.emit(scene_pos)
            self.viewport().update()
            event.accept()
            return

        if self._ruler_mode and event.button() == Qt.MouseButton.LeftButton:
            if img_pos is None:
                # Клик вне допустимой рабочей области изображения
                return

            try:
                mode = self._meas_submode
                if mode == "ruler":
                    self._ruler_dragging = True
                    self._ruler_p1 = img_pos
                    self._ruler_p2 = img_pos
                    self._emit_ruler_info()
                    self.viewport().update()
                    event.accept()
                    return

                elif mode == "disk" or getattr(self, 'active_mode', '') == "moon_disk":
                    img_coords = self.viewport_to_image_coords(event.pos())
                    if img_coords is None:
                        return

                    has_existing_disk = (
                        (self.temp_disk_rect is not None and self.temp_disk_rect.width() > 5.0) or
                        (self.disk_center is not None and self.disk_radius >= 5.0)
                    )
                    if has_existing_disk:
                        handle = self._hit_test_disk_handles(event.pos())
                        if handle in ("tl", "tr", "br", "bl", "edge"):
                            self._disk_transform_mode = "resize"
                            self._disk_active_handle = handle
                            self._disk_drag_start_pos = img_coords
                            rect = self.temp_disk_rect
                            if rect is None:
                                r = self.disk_radius
                                rect = QRectF(self.disk_center.x() - r, self.disk_center.y() - r, 2 * r, 2 * r)
                                self.temp_disk_rect = rect
                            self._disk_drag_start_center = rect.center()
                            self._disk_drag_start_radius = rect.width() / 2.0
                            self.disk_is_dragging = True
                            self.set_status_text("Масштабирование диска: тяните курсор для изменения радиуса.")
                            self.viewport().update()
                            event.accept()
                            return

                        elif handle == "center":
                            self._disk_transform_mode = "move"
                            self._disk_active_handle = "center"
                            self._disk_drag_start_pos = img_coords
                            rect = self.temp_disk_rect
                            if rect is None:
                                r = self.disk_radius
                                rect = QRectF(self.disk_center.x() - r, self.disk_center.y() - r, 2 * r, 2 * r)
                                self.temp_disk_rect = rect
                            self._disk_drag_start_center = rect.center()
                            self._disk_drag_start_radius = rect.width() / 2.0
                            self.disk_is_dragging = True
                            self.set_status_text("Перемещение диска: перетащите диск в новое положение.")
                            self.viewport().update()
                            event.accept()
                            return

                        else:
                            # Клик снаружи диска и маркеров: сброс текущего диска и построение с нуля!
                            self.temp_disk_rect = None
                            self.pending_disk = None
                            self.disk_center = None
                            self.disk_radius = 0.0
                            self._disk_draft_center = None
                            self._disk_draft_radius = 0.0
                            self.disk_points.clear()
                            if self._disk_points is not self.disk_points:
                                self._disk_points.clear()
                            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                                self.measurement_hud.set_disk_prompt("", can_apply=False)

                    if self._disk_method == "center_radius":
                        # Старт построения нового диска с нуля
                        self._disk_transform_mode = "draw"
                        self._disk_active_handle = None
                        self.disk_drag_start = img_coords
                        self.disk_drag_current = self.disk_drag_start
                        self.disk_is_dragging = True
                        self.is_drawing_disk = True
                        self._disk_draft_dragging = True
                        self.temp_disk_rect = None
                        self.pending_disk = None
                        self.disk_center = self.disk_drag_start
                        self.disk_radius = 0.0
                        self.set_status_text("Привязка диска: растяните рамку 1:1 по лимбу Луны")
                        self.viewport().update()
                        event.accept()
                        return

                    elif self._disk_method == "three_points":
                        if len(self.disk_points) >= 3:
                            self.disk_points.clear()
                            if self._disk_points is not self.disk_points:
                                self._disk_points.clear()
                            self.disk_center = None
                            self._disk_draft_center = None
                            self.disk_radius = 0.0
                            self._disk_draft_radius = 0.0
                            self.temp_disk_rect = None
                            self.pending_disk = None

                        # Безопасно добавляем точку в список
                        self.disk_points.append(img_coords)
                        if self._disk_points is not self.disk_points:
                            self._disk_points.append(img_coords)

                        if len(self.disk_points) == 1:
                            self.set_status_text("Точка 1 лимба установлена. Укажите точку 2 на дуге лимба.")
                            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                                self.measurement_hud.set_disk_prompt("Точка 1 лимба установлена", can_apply=False)
                        elif len(self.disk_points) == 2:
                            self.set_status_text("Точка 2 лимба установлена. Укажите точку 3 на дуге лимба.")
                            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                                self.measurement_hud.set_disk_prompt("Точка 2 лимба установлена", can_apply=False)
                        elif len(self.disk_points) == 3:
                            # Строгое условие: вызывать расчет только когда len == 3!
                            res = circle_from_3_points(self.disk_points[0], self.disk_points[1], self.disk_points[2])
                            if res is None:
                                self.set_status_text("Внимание: 3 точки коллинеарны (на одной прямой). Выберите точку 3 в другом месте дуги.")
                                self.disk_points.pop()
                                if self._disk_points and self._disk_points is not self.disk_points:
                                    self._disk_points.pop()
                            else:
                                center, r_disk = res
                                x0, y0 = center.x(), center.y()
                                self.disk_center = center
                                self._disk_draft_center = self.disk_center
                                self.disk_radius = r_disk
                                self._disk_draft_radius = r_disk
                                self.temp_disk_rect = QRectF(x0 - r_disk, y0 - r_disk, 2.0 * r_disk, 2.0 * r_disk)
                                self.pending_disk = (x0, y0, r_disk)
                                self.disk_rect_img = self.temp_disk_rect

                                # Полностью очистить массив 3 точек и сбросить промежуточный результат:
                                self.disk_points.clear()
                                if self._disk_points is not self.disk_points:
                                    self._disk_points.clear()
                                if hasattr(self, 'three_points') and self.three_points is not self.disk_points:
                                    self.three_points.clear()
                                self.three_points_result = None

                                if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                                    self.measurement_hud.set_disk_prompt(f"Окружность построена (R={r_disk:.0f} px)", can_apply=True)
                                self.set_status_text(f"Окружность построена (R={r_disk:.0f} px). Нажмите [✓] или Enter.")

                        self.viewport().update()
                        event.accept()
                        return

                elif mode == "ellipse" or getattr(self, 'active_mode', '') == "crater_ellipse":
                    stage = self.ellipse_stage if self.ellipse_stage in (0, 1, 2, 3) else self._ellipse_step
                    if stage in (0, 3):
                        self.ellipse_stage = 1
                        self._ellipse_step = 1
                        self.ellipse_start = img_pos
                        self._ellipse_p_start = img_pos
                        self.ellipse_end = img_pos
                        self._ellipse_p_end = img_pos
                        self._ellipse_cursor = None
                        self._ellipse_data = None
                        self.ellipse_center = None
                        self.ellipse_a = 0.0
                        self.ellipse_b = 0.0
                        self.ellipse_angle = 0.0
                        if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                            self.measurement_hud.set_ellipse_step(1)
                        self.set_status_text("Шаг 1: Растяните максимальный диаметр кратера вдоль вала.")
                        self.viewport().update()
                        event.accept()
                        return
                    elif stage == 2:
                        self.ellipse_stage = 3
                        self._ellipse_step = 3
                        self._ellipse_cursor = img_pos
                        # Не вызывать математику деления b / a, пока точки не определены и a <= 1e-4
                        p_start = self.ellipse_start if self.ellipse_start is not None else self._ellipse_p_start
                        p_end = self.ellipse_end if self.ellipse_end is not None else self._ellipse_p_end
                        if p_start is not None and p_end is not None:
                            dx = p_end.x() - p_start.x()
                            dy = p_end.y() - p_start.y()
                            if math.hypot(dx, dy) > 1e-4:
                                self._update_ellipse_calc(img_pos)
                        if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                            self.measurement_hud.set_ellipse_step(3)
                        self.set_status_text("Эллипс зафиксирован. Кликните снова для нового замера.")
                        self.viewport().update()
                        event.accept()
                        return

            except Exception:
                traceback.print_exc()
                event.accept()
                return

        if self._profile_mode and event.button() == Qt.MouseButton.LeftButton:
            self._profile_dragging = True
            self._profile_p1 = scene_pos
            self._profile_p2 = scene_pos
            self.viewport().update()
            event.accept()
            return

        if self._crop_mode and self._crop_item is not None:
            r = self._crop_item.crop_rect
            margin = 12 / self.transform().m11()
            tl = r.topLeft()
            tr = r.topRight()
            bl = r.bottomLeft()
            br = r.bottomRight()

            if (scene_pos - tl).manhattanLength() < margin:
                self._crop_drag_mode = 'tl'
            elif (scene_pos - tr).manhattanLength() < margin:
                self._crop_drag_mode = 'tr'
            elif (scene_pos - bl).manhattanLength() < margin:
                self._crop_drag_mode = 'bl'
            elif (scene_pos - br).manhattanLength() < margin:
                self._crop_drag_mode = 'br'
            elif r.contains(scene_pos):
                self._crop_drag_mode = 'move'
            else:
                self._crop_drag_mode = None

            self._crop_drag_start = scene_pos
            self._crop_rect_start = QRectF(r)
            event.accept()
            return

        if self._item.split_active and event.button() == Qt.MouseButton.LeftButton:
            if self._is_near_split_line(scene_pos):
                self._dragging_split = True
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._sync_alt_state(event.modifiers())
        scene_pos = self.mapToScene(event.pos())
        self._mouse_vp = event.position()
        self._emit_pixel_probe(scene_pos)

        if self._loupe_mode or self._alt_pressed:
            self.viewport().update()

        try:
            if self._ruler_mode:
                if (self._meas_submode == "disk" or getattr(self, 'active_mode', '') == "moon_disk") and not self.is_drawing_disk and not self._disk_draft_dragging:
                    handle = self._hit_test_disk_handles(event.pos())
                    if handle != self._disk_hover_handle:
                        self._disk_hover_handle = handle
                        self.setCursor(self._cursor_for_disk_handle(handle))
                        self.viewport().update()

                if self._meas_submode == "ruler" and self._ruler_dragging:
                    self._ruler_p2 = scene_pos
                    self._emit_ruler_info()
                    self.viewport().update()
                    event.accept()
                    return
                elif self._meas_submode == "disk" or getattr(self, 'active_mode', '') == "moon_disk":
                    if self.disk_is_dragging:
                        img_coords = self.viewport_to_image_coords(event.pos())
                        if img_coords is not None:
                            if getattr(self, '_disk_transform_mode', None) == "move" and getattr(self, '_disk_drag_start_pos', None) is not None:
                                delta = img_coords - self._disk_drag_start_pos
                                new_c = self._disk_drag_start_center + delta
                                r = self._disk_drag_start_radius
                                self.disk_center = new_c
                                self._disk_draft_center = new_c
                                self.disk_radius = r
                                self._disk_draft_radius = r
                                self.temp_disk_rect = QRectF(new_c.x() - r, new_c.y() - r, 2 * r, 2 * r)
                                self.pending_disk = (new_c.x(), new_c.y(), r)
                                self.set_status_text(f"Перемещение диска: центр ({new_c.x():.0f}, {new_c.y():.0f})")
                                self.viewport().update()
                                event.accept()
                                return

                            elif getattr(self, '_disk_transform_mode', None) == "resize" and getattr(self, '_disk_drag_start_center', None) is not None:
                                center = self._disk_drag_start_center
                                new_radius = max(5.0, float(max(abs(img_coords.x() - center.x()), abs(img_coords.y() - center.y()))))
                                self.disk_center = center
                                self._disk_draft_center = center
                                self.disk_radius = new_radius
                                self._disk_draft_radius = new_radius
                                self.disk_rect_img = QRectF(center.x() - new_radius, center.y() - new_radius, new_radius * 2.0, new_radius * 2.0)
                                self.pending_disk = (center.x(), center.y(), new_radius)
                                self.set_status_text(f"Масштабирование от центра: R={new_radius:.0f} px")
                                self.viewport().update()
                                event.accept()
                                return

                            elif (self.is_drawing_disk or self._disk_draft_dragging) and self.disk_drag_start is not None:
                                self.disk_drag_current = img_coords
                                p_start = self.disk_drag_start
                                p_cur = self.disk_drag_current

                                cx = (p_start.x() + p_cur.x()) / 2.0
                                cy = (p_start.y() + p_cur.y()) / 2.0
                                center = QPointF(cx, cy)
                                radius = math.hypot(p_cur.x() - p_start.x(), p_cur.y() - p_start.y()) / 2.0

                                self.temp_disk_rect = QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius)
                                self.disk_center = center
                                self.disk_radius = radius
                                self._disk_draft_center = center
                                self._disk_draft_radius = radius
                                self.pending_disk = (cx, cy, radius)
                                self.disk_rect_img = self.temp_disk_rect
                                self.set_status_text(f"Привязка диска по диаметру: D={2.0 * radius:.0f} px, R={radius:.0f} px. Отпустите ЛКМ для фиксации.")
                                self.viewport().update()
                                event.accept()
                                return
                    elif self._disk_method == "three_points" and len(self.disk_points) == 2:
                        self.viewport().update()
                elif self._meas_submode == "ellipse" or getattr(self, 'active_mode', '') == "crater_ellipse":
                    stage = self.ellipse_stage if self.ellipse_stage in (1, 2) else self._ellipse_step
                    if stage == 1:
                        self.ellipse_end = scene_pos
                        self._ellipse_p_end = scene_pos
                        p_start = self.ellipse_start if self.ellipse_start is not None else self._ellipse_p_start
                        if p_start is not None:
                            dx = self.ellipse_end.x() - p_start.x()
                            dy = self.ellipse_end.y() - p_start.y()
                            self.ellipse_a = math.hypot(dx, dy) / 2.0
                            self.ellipse_angle = (math.degrees(math.atan2(dy, dx))) % 360.0
                        self.viewport().update()
                        event.accept()
                        return
                    elif stage == 2:
                        self._ellipse_cursor = scene_pos
                        if self.ellipse_a > 1e-4:
                            self._update_ellipse_calc(scene_pos)
                        self.viewport().update()
                        event.accept()
                        return
        except Exception:
            traceback.print_exc()

        if self._profile_mode and self._profile_dragging:
            self._profile_p2 = scene_pos
            self.profileLineSelected.emit(self._profile_p1, self._profile_p2)
            self.viewport().update()
            event.accept()
            return

        if self._crop_mode and self._crop_item is not None:
            r = self._crop_item.crop_rect
            if self._crop_drag_mode is not None:
                delta = scene_pos - self._crop_drag_start
                r_start = self._crop_rect_start
                new_r = QRectF(r_start)

                if self._crop_drag_mode == 'move':
                    new_r.translate(delta.x(), delta.y())
                elif self._crop_drag_mode == 'tl':
                    new_r.setTopLeft(r_start.topLeft() + delta)
                elif self._crop_drag_mode == 'tr':
                    new_r.setTopRight(r_start.topRight() + delta)
                elif self._crop_drag_mode == 'bl':
                    new_r.setBottomLeft(r_start.bottomLeft() + delta)
                elif self._crop_drag_mode == 'br':
                    new_r.setBottomRight(r_start.bottomRight() + delta)

                new_r = new_r.normalized()
                new_r.setLeft(max(0.0, new_r.left()))
                new_r.setTop(max(0.0, new_r.top()))
                new_r.setRight(min(self._crop_item.img_w, new_r.right()))
                new_r.setBottom(min(self._crop_item.img_h, new_r.bottom()))

                if new_r.width() > 10 and new_r.height() > 10:
                    self._crop_item.crop_rect = new_r
                    self._crop_item.update()
                event.accept()
                return

            margin = 12 / self.transform().m11()
            if (scene_pos - r.topLeft()).manhattanLength() < margin or (scene_pos - r.bottomRight()).manhattanLength() < margin:
                self.viewport().setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif (scene_pos - r.topRight()).manhattanLength() < margin or (scene_pos - r.bottomLeft()).manhattanLength() < margin:
                self.viewport().setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif r.contains(scene_pos):
                self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            super().mouseMoveEvent(event)
            return

        if self._dragging_split:
            if self._item.split_orientation == "Vertical":
                w = self._item.processed_pixmap.width()
                if w > 0:
                    frac = scene_pos.x() / w
                    self._item.split_fraction = float(np.clip(frac, 0.01, 0.99))
                    self._item.update()
                    self.viewport().update()
            else:
                h = self._item.processed_pixmap.height()
                if h > 0:
                    frac = scene_pos.y() / h
                    self._item.split_fraction = float(np.clip(frac, 0.01, 0.99))
                    self._item.update()
                    self.viewport().update()
            event.accept()
            return

        if self._item.split_active and self._is_near_split_line(scene_pos):
            cursor = Qt.CursorShape.SplitHCursor if self._item.split_orientation == "Vertical" else Qt.CursorShape.SplitVCursor
            self.viewport().setCursor(cursor)
        elif self._ruler_mode and (self._meas_submode == "disk" or getattr(self, 'active_mode', '') == "moon_disk") and not self.disk_is_dragging:
            has_existing = (
                self.temp_disk_rect is not None and self.temp_disk_rect.width() > 5.0 and
                self.disk_center is not None and self.disk_radius >= 5.0
            )
            if has_existing:
                view_scale = max(1e-4, self.transform().m11())
                c_vp = self.mapFromScene(self.disk_center)
                r_vp = self.disk_radius * view_scale
                vp_pos = event.pos()
                d_vp = math.hypot(vp_pos.x() - c_vp.x(), vp_pos.y() - c_vp.y())

                tol_corner = 12.0
                tl_vp = QPointF(c_vp.x() - r_vp, c_vp.y() - r_vp)
                tr_vp = QPointF(c_vp.x() + r_vp, c_vp.y() - r_vp)
                br_vp = QPointF(c_vp.x() + r_vp, c_vp.y() + r_vp)
                bl_vp = QPointF(c_vp.x() - r_vp, c_vp.y() + r_vp)

                if math.hypot(vp_pos.x() - tl_vp.x(), vp_pos.y() - tl_vp.y()) <= tol_corner or math.hypot(vp_pos.x() - br_vp.x(), vp_pos.y() - br_vp.y()) <= tol_corner:
                    self.viewport().setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif math.hypot(vp_pos.x() - tr_vp.x(), vp_pos.y() - tr_vp.y()) <= tol_corner or math.hypot(vp_pos.x() - bl_vp.x(), vp_pos.y() - bl_vp.y()) <= tol_corner:
                    self.viewport().setCursor(Qt.CursorShape.SizeBDiagCursor)
                elif abs(d_vp - r_vp) <= 10.0:
                    dx = vp_pos.x() - c_vp.x()
                    dy = vp_pos.y() - c_vp.y()
                    ang = abs(math.degrees(math.atan2(dy, dx)))
                    if ang <= 25 or ang >= 155:
                        self.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
                    elif 65 <= ang <= 115:
                        self.viewport().setCursor(Qt.CursorShape.SizeVerCursor)
                    elif (25 < ang < 65 and dx * dy > 0) or (115 < ang < 155 and dx * dy > 0):
                        self.viewport().setCursor(Qt.CursorShape.SizeFDiagCursor)
                    else:
                        self.viewport().setCursor(Qt.CursorShape.SizeBDiagCursor)
                elif d_vp < r_vp - 10.0:
                    self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        elif self._any_tool_active():
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag else Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        try:
            if self._ruler_mode:
                if self._meas_submode == "ruler" and self._ruler_dragging:
                    self._ruler_dragging = False
                    self._ruler_p2 = scene_pos
                    self._emit_ruler_info()
                    self.viewport().update()
                    event.accept()
                    return
                elif self._meas_submode == "disk" or getattr(self, 'active_mode', '') == "moon_disk":
                    if self.disk_is_dragging or self.is_drawing_disk or self._disk_draft_dragging:
                        self.disk_is_dragging = False
                        self.is_drawing_disk = False
                        self._disk_draft_dragging = False
                        self._disk_draft_drag_target = None
                        self._disk_active_handle = None
                        self._disk_transform_mode = None
                        if self.temp_disk_rect and self.temp_disk_rect.width() > 5:
                            c = self.temp_disk_rect.center()
                            r = self.temp_disk_rect.width() / 2.0
                            self.pending_disk = (c.x(), c.y(), r)
                            self.disk_center = c
                            self.disk_radius = r
                            self._disk_draft_center = c
                            self._disk_draft_radius = r
                            self.set_status_text(f"Диск построен (R={r:.0f} px). Нажмите [✓] или Enter.")
                            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                                self.measurement_hud.set_disk_prompt(f"Диск построен (R={r:.0f} px)", can_apply=True)
                        elif self.disk_center is not None and self.disk_radius > 5:
                            c = self.disk_center
                            r = self.disk_radius
                            self.pending_disk = (c.x(), c.y(), r)
                            self.temp_disk_rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
                            self.set_status_text(f"Диск построен (R={r:.0f} px). Нажмите [✓] или Enter.")
                            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                                self.measurement_hud.set_disk_prompt(f"Диск построен (R={r:.0f} px)", can_apply=True)
                        else:
                            self.temp_disk_rect = None
                            self.pending_disk = None
                            self.disk_center = None
                            self.disk_radius = 0.0
                            self._disk_draft_center = None
                            self._disk_draft_radius = 0.0
                            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                                self.measurement_hud.set_disk_prompt("", can_apply=False)
                            self.disk_radius = 0.0
                            self._disk_draft_center = None
                            self._disk_draft_radius = 0.0
                        self.viewport().update()
                        event.accept()
                        return
                elif self._meas_submode == "ellipse" or getattr(self, 'active_mode', '') == "crater_ellipse":
                    stage = self.ellipse_stage if self.ellipse_stage in (1, 2) else self._ellipse_step
                    if stage == 1:
                        p_start = self.ellipse_start if self.ellipse_start is not None else self._ellipse_p_start
                        p_end = self.ellipse_end if self.ellipse_end is not None else self._ellipse_p_end
                        if p_start is not None and p_end is not None:
                            dx = p_end.x() - p_start.x()
                            dy = p_end.y() - p_start.y()
                            dist = np.hypot(dx, dy)
                            if dist >= 3.0:
                                self.ellipse_stage = 2
                                self._ellipse_step = 2
                                self.ellipse_a = dist / 2.0
                                self._ellipse_cursor = scene_pos
                                self._update_ellipse_calc(scene_pos)
                                if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                                    self.measurement_hud.set_ellipse_step(2)
                                self.set_status_text("Шаг 2: Перемещайте курсор для задания малой оси. Кликните ЛКМ для фиксации.")
                            else:
                                self.ellipse_stage = 0
                                self._ellipse_step = 0
                                self.ellipse_start = None
                                self._ellipse_p_start = None
                                self.ellipse_end = None
                                self._ellipse_p_end = None
                                self.ellipse_a = 0.0
                                self.ellipse_b = 0.0
                                if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                                    self.measurement_hud.set_ellipse_step(1)
                        self.viewport().update()
                        event.accept()
                        return
        except Exception:
            traceback.print_exc()

        if self._profile_mode and self._profile_dragging:
            self._profile_dragging = False
            self.profileLineSelected.emit(self._profile_p1, self._profile_p2)
            event.accept()
            return

        if self._crop_mode:
            self._crop_drag_mode = None
            event.accept()
            return

        if self._dragging_split and event.button() == Qt.MouseButton.LeftButton:
            self._dragging_split = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._ruler_mode and self._meas_submode == "disk" and event.button() == Qt.MouseButton.LeftButton:
            self.apply_disk_alignment()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Alt:
            self._alt_pressed = True
            self.viewport().update()

        if event.key() == Qt.Key.Key_Escape:
            if getattr(self, 'clipping_overlay_mode', None) is not None:
                self.set_clipping_overlay_mode(None)
                event.accept()
                return

            if self._ruler_mode:
                if (self._meas_submode == "disk" or getattr(self, 'active_mode', '') == "moon_disk") and (
                    self.disk_center is not None or self.temp_disk_rect is not None or self.pending_disk is not None or self._disk_draft_center is not None or self.disk_points or self._disk_points
                ):
                    self.disk_center = None
                    self._disk_draft_center = None
                    self.disk_radius = 0.0
                    self._disk_draft_radius = 0.0
                    self.temp_disk_rect = None
                    self.pending_disk = None
                    self.disk_is_dragging = False
                    self.disk_drag_start = None
                    self.disk_drag_current = None
                    self._disk_state = "idle"
                    self._disk_active_handle = None
                    self._disk_hover_handle = None
                    self._disk_draw_start = None
                    self._disk_anchor = None
                    self.disk_points.clear()
                    if self._disk_points is not self.disk_points:
                        self._disk_points.clear()
                    self.is_drawing_disk = False
                    self._disk_draft_dragging = False
                    self.set_status_text("Черновик привязки диска сброшен.")
                    self.viewport().update()
                    event.accept()
                    return
                elif (self._meas_submode == "ellipse" or getattr(self, 'active_mode', '') == "crater_ellipse") and (
                    self.ellipse_stage > 0 or self._ellipse_step > 0
                ):
                    self.ellipse_stage = 0
                    self._ellipse_step = 0
                    self.ellipse_start = None
                    self._ellipse_p_start = None
                    self.ellipse_end = None
                    self._ellipse_p_end = None
                    self._ellipse_cursor = None
                    self._ellipse_data = None
                    self.ellipse_center = None
                    self.ellipse_a = 0.0
                    self.ellipse_b = 0.0
                    self.ellipse_angle = 0.0
                    if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                        self.measurement_hud.clear_results()
                    self.set_status_text("Замер эллипса кратера отменён.")
                    self.viewport().update()
                    event.accept()
                    return

            self.reset_to_default_tool()
            self.clear_ruler()
            self.clear_psf_marker()
            self.cropCancelled.emit()
            event.accept()
            return

        if self._ruler_mode and (self._meas_submode == "disk" or getattr(self, 'active_mode', '') == "moon_disk"):
            center = self.disk_center if self.disk_center is not None else self._disk_draft_center
            radius = self.disk_radius if self.disk_radius > 0.0 else self._disk_draft_radius

            if center is not None and radius >= 5.0:
                shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                move_step = 10.0 if shift else 1.0
                rad_step = 5.0 if shift else 1.0

                if event.key() == Qt.Key.Key_Left:
                    self.disk_center = QPointF(center.x() - move_step, center.y())
                    self._disk_draft_center = self.disk_center
                    self.pending_disk = (self.disk_center.x(), self.disk_center.y(), radius)
                    self.temp_disk_rect = QRectF(self.disk_center.x() - radius, self.disk_center.y() - radius, 2 * radius, 2 * radius)
                    self.viewport().update()
                    event.accept()
                    return
                elif event.key() == Qt.Key.Key_Right:
                    self.disk_center = QPointF(center.x() + move_step, center.y())
                    self._disk_draft_center = self.disk_center
                    self.pending_disk = (self.disk_center.x(), self.disk_center.y(), radius)
                    self.temp_disk_rect = QRectF(self.disk_center.x() - radius, self.disk_center.y() - radius, 2 * radius, 2 * radius)
                    self.viewport().update()
                    event.accept()
                    return
                elif event.key() == Qt.Key.Key_Up:
                    self.disk_center = QPointF(center.x(), center.y() - move_step)
                    self._disk_draft_center = self.disk_center
                    self.pending_disk = (self.disk_center.x(), self.disk_center.y(), radius)
                    self.temp_disk_rect = QRectF(self.disk_center.x() - radius, self.disk_center.y() - radius, 2 * radius, 2 * radius)
                    self.viewport().update()
                    event.accept()
                    return
                elif event.key() == Qt.Key.Key_Down:
                    self.disk_center = QPointF(center.x(), center.y() + move_step)
                    self._disk_draft_center = self.disk_center
                    self.pending_disk = (self.disk_center.x(), self.disk_center.y(), radius)
                    self.temp_disk_rect = QRectF(self.disk_center.x() - radius, self.disk_center.y() - radius, 2 * radius, 2 * radius)
                    self.viewport().update()
                    event.accept()
                    return
                elif event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                    new_r = radius + rad_step
                    self.disk_radius = new_r
                    self._disk_draft_radius = new_r
                    self.pending_disk = (center.x(), center.y(), new_r)
                    self.temp_disk_rect = QRectF(center.x() - new_r, center.y() - new_r, 2 * new_r, 2 * new_r)
                    self.set_status_text(f"Радиус диска: {new_r:.0f} px")
                    self.viewport().update()
                    event.accept()
                    return
                elif event.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
                    new_r = max(5.0, radius - rad_step)
                    self.disk_radius = new_r
                    self._disk_draft_radius = new_r
                    self.pending_disk = (center.x(), center.y(), new_r)
                    self.temp_disk_rect = QRectF(center.x() - new_r, center.y() - new_r, 2 * new_r, 2 * new_r)
                    self.set_status_text(f"Радиус диска: {new_r:.0f} px")
                    self.viewport().update()
                    event.accept()
                    return

        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._ruler_mode and (self._meas_submode == "disk" or getattr(self, 'active_mode', '') == "moon_disk"):
                radius = self.disk_radius if self.disk_radius > 0.0 else self._disk_draft_radius
                pts = self.disk_points if len(self.disk_points) > 0 else self._disk_points
                if radius >= 5.0 or len(pts) == 3:
                    self.apply_disk_alignment()
                    event.accept()
                    return
            if self._crop_mode:
                self.apply_current_crop()
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Alt:
            self._alt_pressed = False
            self.viewport().update()
        super().keyReleaseEvent(event)

    def wheelEvent(self, event):
        center = self.disk_center if self.disk_center is not None else self._disk_draft_center
        radius = self.disk_radius if self.disk_radius > 0.0 else self._disk_draft_radius
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if self._ruler_mode and (self._meas_submode == "disk" or getattr(self, 'active_mode', '') == "moon_disk") and center is not None and radius >= 5.0 and ctrl:
            delta = 1.0 if event.angleDelta().y() > 0 else -1.0
            new_r = max(5.0, radius + delta * 1.0)
            self.disk_radius = new_r
            self._disk_draft_radius = new_r
            self.set_status_text(f"Радиус диска: {new_r:.1f} px")
            if hasattr(self, 'measurement_hud') and self.measurement_hud is not None:
                self.measurement_hud.set_disk_prompt(f"Диск R={new_r:.0f} px. Enter — применить", can_apply=True)
            self.viewport().update()
            event.accept()
            return

        factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        self.scale(factor, factor)
        self.zoomChanged.emit(self.get_current_zoom_str())
        self.viewport().update()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.pixelProbeChanged.emit("X: ---, Y: ---", "")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            image_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.fit', '.fits')
            for url in event.mimeData().urls():
                fp = url.toLocalFile()
                if os.path.splitext(fp)[1].lower() in image_exts:
                    event.acceptProposedAction()
                    if hasattr(self, 'welcome_widget') and self.welcome_widget.isVisible():
                        self.welcome_widget.set_drag_active(True)
                    return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        if hasattr(self, 'welcome_widget') and self.welcome_widget.isVisible():
            self.welcome_widget.set_drag_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if hasattr(self, 'welcome_widget') and self.welcome_widget.isVisible():
            self.welcome_widget.set_drag_active(False)
        image_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.fit', '.fits')
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                ext = os.path.splitext(file_path)[1].lower()
                if ext in image_exts:
                    event.acceptProposedAction()
                    self.fileDropped.emit(file_path)
                    return
        super().dropEvent(event)
