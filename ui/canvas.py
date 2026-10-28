#"""Холст AstroCanvas: интерактивный Split View, лупа 300% (Alt), кадрирование и профилометр."""
import os
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
    fileDropped = pyqtSignal(str)
    openFileDialogRequested = pyqtSignal()
    pixelProbeChanged = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._item = SplitImageGraphicsItem()
        self._scene.addItem(self._item)
        self._current_frame_float = None

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
        self._ruler_unit = "all"

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

    def reset_for_new_image(self):
        """Полный сброс состояния холста, оверлеев и буферов перед загрузкой нового файла."""
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

    def set_ruler_mode(self, active: bool):
        self._ruler_mode = active
        if not active:
            self._ruler_dragging = False
        self._update_interaction_state()

    def set_telescope_params(self, focal_len_mm: float, pixel_size_um: float):
        self._telescope_focal_len = max(1.0, float(focal_len_mm))
        self._telescope_pixel_size = max(0.1, float(pixel_size_um))
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

        km = ang_arcsec * 1.8636

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

        dist_str, _ = self._format_ruler_dist(dist_px)
        summary = f"L: {dist_str} | Angle: {angle_deg:.1f}°"
        self.rulerMeasured.emit(dist_px, angle_deg, summary)
        self.set_status_text(f"Ruler: {summary}")

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

        # 2.2. Линейка расстояний и углов (Pixel / Angular Ruler)
        if (self._ruler_mode or self._ruler_dragging) and self._ruler_p1 is not None and self._ruler_p2 is not None:
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
            b_rect = QRectF(badge_x - bw / 2.0, badge_y - bh / 2.0, bw, bh)

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
        scene_pos = self.mapToScene(event.pos())

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
            self._ruler_dragging = True
            self._ruler_p1 = scene_pos
            self._ruler_p2 = scene_pos
            self._emit_ruler_info()
            self.viewport().update()
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

        if self._ruler_mode and self._ruler_dragging:
            self._ruler_p2 = scene_pos
            self._emit_ruler_info()
            self.viewport().update()
            event.accept()
            return

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
        elif self._any_tool_active():
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag else Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._ruler_mode and self._ruler_dragging:
            self._ruler_dragging = False
            self._ruler_p2 = scene_pos if 'scene_pos' in locals() else self.mapToScene(event.pos())
            self._emit_ruler_info()
            self.viewport().update()
            event.accept()
            return

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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Alt:
            self._alt_pressed = True
            self.viewport().update()

        if event.key() == Qt.Key.Key_Escape:
            self.reset_to_default_tool()
            self.clear_ruler()
            self.clear_psf_marker()
            self.cropCancelled.emit()
            event.accept()
            return

        if self._crop_mode:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
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
