"""Модуль пакетной обработки кадров (Batch Processing Pipeline) для AstroRack DAW.

Содержит:
- Изолированную математическую функцию обработки кадра process_frame_pipeline (без зависимостей от PyQt UI).
- Вспомогательные функции детекции ROI планеты, безопасного чтения и записи 16-бит/8-бит кадров.
- Фоновый воркер BatchWorker (QThread).
- Диалог конфигурации пакетной обработки BatchSetupDialog.
- Диалог хода выполнения и статуса BatchProgressDialog.
"""
import os
import sys
import time
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QRadioButton, QButtonGroup, QLineEdit,
    QComboBox, QListWidget, QFileDialog, QFrame, QMessageBox,
    QWidget, QCheckBox, QAbstractItemView
)
from PyQt6.QtGui import QIcon, QDesktopServices

from core.processing import AstroEngine, soft_clip_limiter
from core.enhance import apply_local_contrast, apply_color_engine
from core.pipeline import blend_dry_wet, _extract_wet_ratio
from core.sound import sound_engine
from utils.file_io import get_resource_path, load_image


# =============================================================================
# ИЗОЛИРОВАННАЯ МАТЕМАТИЧЕСКАЯ ОБРАБОТКА КАДРА (ЧИСТАЯ ФУНКЦИЯ)
# =============================================================================

def detect_planetary_roi(img: np.ndarray, margin_ratio: float = 0.15) -> tuple[int, int, int, int]:
    """Автоматически находит ограничивающий прямоугольник (x1, y1, x2, y2) диска планеты.

    Использует бинаризацию Оцу по зеленому каналу/яркости, находит контур планеты
    и добавляет относительный отступ margin_ratio.
    Возвращает (x1, y1, x2, y2) в пикселях.
    """
    h, w = img.shape[:2]
    if img.ndim == 3 and img.shape[2] >= 3:
        # Зеленый канал обладает максимальным отношением сигнал/шум
        gray = img[:, :, 1]
    elif img.ndim == 3:
        gray = img[:, :, 0]
    else:
        gray = img

    gray_u8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    thresh_val, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thresh_val < 15:
        _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, w, h

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 25:
        return 0, 0, w, h

    bx, by, bw, bh = cv2.boundingRect(c)
    pad_x = int(bw * margin_ratio)
    pad_y = int(bh * margin_ratio)

    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(w, bx + bw + pad_x)
    y2 = min(h, by + bh + pad_y)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return 0, 0, w, h

    return x1, y1, x2, y2


def apply_fixed_rgb_shifts(img: np.ndarray, dx_r: float, dy_r: float, dx_b: float, dy_b: float) -> np.ndarray:
    """Применяет заданные фиксированные сдвиги R и B каналов относительно G."""
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    h, w = img.shape[:2]
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)

    m_r = np.array([[1.0, 0.0, dx_r], [0.0, 1.0, dy_r]], dtype=np.float32)
    m_b = np.array([[1.0, 0.0, dx_b], [0.0, 1.0, dy_b]], dtype=np.float32)

    r_aligned = cv2.warpAffine(r, m_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b_aligned = cv2.warpAffine(b, m_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    aligned = np.stack([r_aligned, g, b_aligned], axis=-1)
    return np.clip(aligned, 0.0, 1.0)


def process_frame_pipeline(
    raw_frame: np.ndarray,
    rack_state: dict,
    options: dict = None,
    cancel_check = None
) -> np.ndarray:
    """Изолированная функция обработки отдельного астрономического кадра.

    Параметры:
        raw_frame: numpy.ndarray изображения (uint16, uint8 или float32 [0..1], ч/б или RGB).
        rack_state: сериализованный словарь параметров рэка AstroRack DAW
                    (вейвлеты, RL-деконволюция, тоновая кривая, моно/RGB, локальный контраст, цвет).
        options: словарь доп. параметров пакетной обработки:
                 - 'auto_crop_roi': bool (авто-кадрирование по диску планеты)
                 - 'reference_crop_rect': tuple (x1, y1, x2, y2) фиксированного ROI
                 - 'auto_rgb_per_frame': bool (индивидуальный фазовый расчет сдвига на каждом кадре)
                 - 'fixed_shifts': tuple (dx_r, dy_r, dx_b, dy_b) для фиксированного выравнивания
        cancel_check: функция без аргументов, возвращающая True при запросе прерывания.

    Возвращает:
        numpy.ndarray float32 в диапазоне [0.0, 1.0] с обработанным кадром, либо None при отмене.
    """
    if raw_frame is None or raw_frame.size == 0:
        return None

    if cancel_check and cancel_check():
        return None

    options = options or {}

    # 1. Приведение типа данных к float32 [0.0, 1.0] с сохранением разрядности
    if raw_frame.dtype == np.uint16:
        img = raw_frame.astype(np.float32) / 65535.0
    elif raw_frame.dtype == np.uint8:
        img = raw_frame.astype(np.float32) / 255.0
    else:
        img = np.clip(raw_frame.astype(np.float32), 0.0, 1.0)

    # 2. Кадрирование по Planetary ROI (если запрошено в опциях)
    if options.get('auto_crop_roi', False):
        ref_rect = options.get('reference_crop_rect', None)
        if ref_rect is not None:
            rx1, ry1, rx2, ry2 = ref_rect
            h, w = img.shape[:2]
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
            if (rx2 - rx1) >= 10 and (ry2 - ry1) >= 10:
                img = img[ry1:ry2, rx1:rx2].copy()
        else:
            px1, py1, px2, py2 = detect_planetary_roi(img)
            if (px2 - px1) < img.shape[1] or (py2 - py1) < img.shape[0]:
                img = img[py1:py2, px1:px2].copy()

    # 3. Master Bypass: если глобальный байпас включен, эффекты не применяются
    if rack_state.get('global_bypass', False):
        return np.ascontiguousarray(img, dtype=np.float32)

    # =========================================================================
    # СТАДИЯ 1 (Insert 4): Auto RGB Align & Extract Green Channel
    # =========================================================================
    ins4_on = bool(rack_state.get('insert4_enabled', False))
    do_align = ins4_on and bool(rack_state.get('auto_rgb_align', False))
    do_green = ins4_on and bool(rack_state.get('extract_green', False))

    if ins4_on and img.ndim == 3 and img.shape[2] == 3:
        if do_align:
            auto_rgb_per_frame = options.get('auto_rgb_per_frame', True)
            if auto_rgb_per_frame:
                if cancel_check and cancel_check():
                    return None
                aligned, _, _ = AstroEngine.auto_rgb_align(img)
                img = np.ascontiguousarray(aligned, dtype=np.float32)
            else:
                shifts = options.get('fixed_shifts', (0.0, 0.0, 0.0, 0.0))
                img = apply_fixed_rgb_shifts(img, *shifts)

        if cancel_check and cancel_check():
            return None

        if do_green:
            img = np.ascontiguousarray(AstroEngine.extract_green_channel(img), dtype=np.float32)

    if cancel_check and cancel_check():
        return None

    # =========================================================================
    # СТАДИЯ 2 (Insert 1): Wavelet (à trous) / Unsharp Mask
    # =========================================================================
    ins1_on = bool(rack_state.get('insert1_enabled', False))
    w_mode = int(rack_state.get('w_mode', 0))
    w_l1 = round(float(rack_state.get('w_l1', 1.0)), 4)
    w_l2 = round(float(rack_state.get('w_l2', 1.0)), 4)
    w_denoise = round(float(rack_state.get('w_denoise', 0.0)), 4)
    w_wet_ratio = _extract_wet_ratio(rack_state, 'w_wet_mix', 'insert1_wet_mix', 'wet_mix', default=1.0)

    w_layers = None
    if w_mode == 0:
        w_layers = AstroEngine.trim_neutral_layers(
            AstroEngine.normalize_layers_config(
                rack_state.get('layers_config'), w_l1, w_l2, w_denoise
            )
        )
        ins1_neutral = not ins1_on or (
            len(w_layers) == 1 and abs(w_layers[0]["amount"] - 1.0) < 1e-4 and w_layers[0]["denoise"] <= 1e-5
        )
    else:
        ins1_neutral = not ins1_on or abs(w_l1 - 1.0) < 1e-4

    if not ins1_neutral and w_wet_ratio > 0.001:
        if cancel_check and cancel_check():
            return None

        if w_mode == 0:
            n_layers = len(w_layers)
            decomp = AstroEngine.atrous_decompose(img, n_layers, cancel_check=cancel_check)
            if decomp is None:
                return None
            residual, details = decomp
            stage1_wet = AstroEngine.atrous_recompose(residual, details, w_layers)
            stage1_wet = np.clip(stage1_wet, 0.0, 1.0, out=stage1_wet)
        else:
            if img.ndim == 2:
                blurred = gaussian_filter(img, sigma=w_l2, mode='reflect')
            else:
                planes = []
                for c in range(img.shape[2]):
                    if cancel_check and cancel_check():
                        return None
                    planes.append(gaussian_filter(img[:, :, c], sigma=w_l2, mode='reflect'))
                blurred = np.stack(planes, axis=-1)

            diff = img - blurred
            if w_denoise > 0.0:
                thresh = np.float32(w_denoise)
                diff = np.sign(diff) * np.maximum(np.abs(diff) - thresh, np.float32(0.0))

            stage1_wet = np.clip(img + np.float32(w_l1 - 1.0) * diff, 0.0, 1.0)

        img = blend_dry_wet(img, stage1_wet, w_wet_ratio)

    if cancel_check and cancel_check():
        return None

    # =========================================================================
    # СТАДИЯ 3 (Insert 2): Richardson-Lucy Deconvolution
    # =========================================================================
    ins2_on = bool(rack_state.get('insert2_enabled', False))
    rl_sigma = round(float(rack_state.get('rl_sigma', 1.0)), 3)
    rl_iters = int(rack_state.get('rl_iters', 0))
    rl_anti_ringing = bool(rack_state.get('rl_anti_ringing', False))
    rl_wet_ratio = _extract_wet_ratio(rack_state, 'rl_wet_mix', 'insert2_wet_mix', 'wet_mix', default=1.0)

    if ins2_on and rl_iters > 0 and rl_wet_ratio > 0.001:
        estimate = np.maximum(img.copy(), np.float32(1e-6))
        sigma_axes = (rl_sigma, rl_sigma) if img.ndim == 2 else (rl_sigma, rl_sigma, 0.0)

        for _ in range(rl_iters):
            if cancel_check and cancel_check():
                return None
            conv = gaussian_filter(estimate, sigma=sigma_axes, mode='reflect')
            np.maximum(conv, np.float32(1e-12), out=conv)
            relative_blur = img / conv
            if cancel_check and cancel_check():
                return None
            corr = gaussian_filter(relative_blur, sigma=sigma_axes, mode='reflect')
            estimate *= corr

        if rl_anti_ringing:
            if img.ndim == 2:
                gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
                grad = np.hypot(gx, gy)
                edge_mask = np.clip(gaussian_filter(grad, sigma=2.0) * 4.0, 0.0, 1.0).astype(np.float32)
            else:
                masks = []
                for c in range(img.shape[2]):
                    if cancel_check and cancel_check():
                        return None
                    plane = img[:, :, c]
                    gx = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
                    grad = np.hypot(gx, gy)
                    masks.append(np.clip(gaussian_filter(grad, sigma=2.0) * 4.0, 0.0, 1.0).astype(np.float32))
                edge_mask = np.stack(masks, axis=-1)

            weight = edge_mask * np.float32(0.85)
            stage2_wet = np.clip(estimate * (np.float32(1.0) - weight) + img * weight, 0.0, 1.0)
        else:
            stage2_wet = np.clip(estimate, 0.0, 1.0)

        img = blend_dry_wet(img, stage2_wet, rl_wet_ratio)

    if cancel_check and cancel_check():
        return None

    # =========================================================================
    # СТАДИЯ 4 (Insert 3): Tone & Contrast (Curves)
    # =========================================================================
    ins3_on = bool(rack_state.get('insert3_enabled', False))
    tc_black = round(float(rack_state.get('tc_black', 0.0)), 4)
    tc_gamma = round(float(rack_state.get('tc_gamma', 1.0)), 4)
    tc_contrast = round(float(rack_state.get('tc_contrast', 1.0)), 4)

    ins3_neutral = (
        not ins3_on
        or (tc_black <= 1e-5 and abs(tc_gamma - 1.0) <= 1e-3 and abs(tc_contrast - 1.0) <= 1e-3)
    )
    if not ins3_neutral:
        if cancel_check and cancel_check():
            return None
        # Быстрый точный 16-бит LUT
        lut_in = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
        lut_out = AstroEngine.tone_and_contrast(lut_in, tc_black, tc_gamma, tc_contrast).astype(np.float32)
        idx = np.clip(img * np.float32(65535.0), 0.0, 65535.0).astype(np.uint16)
        img = lut_out[idx]

    if cancel_check and cancel_check():
        return None

    # =========================================================================
    # СТАДИЯ 5 (Insert 5): Local Contrast / Clarity (CLAHE / High-Pass)
    # =========================================================================
    ins5_on = bool(rack_state.get('insert5_enabled', False))
    lc_method = str(rack_state.get('lc_method', 'clahe')).lower()
    lc_is_hp = lc_method in ('high_pass', 'highpass', 'hp')
    lc_amount = round(float(rack_state.get('lc_amount', 0.0 if lc_is_hp else 1.0)), 3)
    lc_radius = round(float(rack_state.get('lc_radius', 30.0 if lc_is_hp else 16.0)), 2)
    lc_wet_ratio = _extract_wet_ratio(rack_state, 'lc_wet_mix', 'insert5_wet_mix', 'wet_mix', default=1.0)

    ins5_neutral = (not ins5_on) or (lc_amount <= 1e-3 if lc_is_hp else lc_amount <= 1.0 + 1e-3)
    if not ins5_neutral and lc_wet_ratio > 0.001:
        if cancel_check and cancel_check():
            return None
        stage5_wet = apply_local_contrast(
            img, lc_amount, lc_radius, 'high_pass' if lc_is_hp else 'clahe'
        )
        img = blend_dry_wet(img, stage5_wet, lc_wet_ratio)

    if cancel_check and cancel_check():
        return None

    # =========================================================================
    # СТАДИЯ 6 (Insert 6): Color Engine & Background Neutralization
    # =========================================================================
    ins6_on = bool(rack_state.get('insert6_enabled', False))
    cs_sat = round(float(rack_state.get('cs_saturation', 1.0)), 3)
    cs_mineral = round(float(rack_state.get('cs_mineral', 0.0)), 3)
    cs_bg = bool(rack_state.get('cs_bg_neutralize', False))
    cs_coords = rack_state.get('cs_bg_coords', None)
    cs_coords = tuple(float(v) for v in cs_coords) if cs_coords is not None else None
    cs_wet_ratio = _extract_wet_ratio(rack_state, 'cs_wet_mix', 'insert6_wet_mix', 'wet_mix', default=1.0)

    ins6_neutral = (
        (not ins6_on)
        or img.ndim != 3 or img.shape[2] < 3
        or (abs(cs_sat - 1.0) <= 1e-3 and cs_mineral <= 1e-3 and not cs_bg)
    )
    if not ins6_neutral and cs_wet_ratio > 0.001:
        if cancel_check and cancel_check():
            return None
        stage6_wet = apply_color_engine(img, cs_sat, cs_mineral, cs_bg, cs_coords)
        img = blend_dry_wet(img, stage6_wet, cs_wet_ratio)

    limiter_on = bool(rack_state.get('master_limiter_enabled', False) or rack_state.get('limiter_enabled', False))
    if limiter_on:
        limiter_thresh = float(rack_state.get('limiter_threshold', 0.88))
        img = soft_clip_limiter(img, threshold=limiter_thresh)

    return np.ascontiguousarray(np.clip(img, 0.0, 1.0), dtype=np.float32)


# =============================================================================
# БЕЗОПАСНЫЙ ВВОД / ВЫВОД ИЗОБРАЖЕНИЙ (16-BIT / 8-BIT)
# =============================================================================

def read_image_file(file_path: str) -> tuple[np.ndarray, np.dtype, bool]:
    """Безопасное чтение изображения (16-бит TIFF/PNG/FITS) с поддержкой кириллицы.

    Возвращает:
        (raw_array_in_rgb, original_dtype, is_color)
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.fit', '.fits'):
        data = load_image(file_path)
        return data, np.float32, data.ndim == 3

    buf = np.fromfile(file_path, dtype=np.uint8)
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Не удалось декодировать файл изображения: {file_path}")

    orig_dtype = raw.dtype
    is_color = (raw.ndim == 3 and raw.shape[2] >= 3)
    if is_color:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    return raw, orig_dtype, is_color


def save_processed_frame(
    dest_path: str,
    img_float: np.ndarray,
    format_choice: str = "16-bit TIFF (Uncompressed)",
    orig_dtype: np.dtype = np.uint16
) -> bool:
    """Сохраняет изображение float32 [0.0, 1.0] с контролем разрядности и форматов.

    - 16-bit TIFF: uint16 без потерь и сжатия.
    - 8-bit PNG: uint8 [0..255].
    - Исходный формат: подстраивается под тип и расширение источника.
    """
    clamped = np.clip(img_float, 0.0, 1.0)
    ext = os.path.splitext(dest_path)[1].lower()

    if "8-bit" in format_choice or (format_choice == "Исходный формат" and orig_dtype == np.uint8 and ext in ('.png', '.jpg', '.jpeg')):
        export_data = (clamped * 255.0).astype(np.uint8)
        if export_data.ndim == 3 and export_data.shape[2] == 3:
            export_data = cv2.cvtColor(export_data, cv2.COLOR_RGB2BGR)
        save_ext = ext if ext else '.png'
        success, buf = cv2.imencode(save_ext, export_data)
    else:
        export_data = (clamped * 65535.0).astype(np.uint16)
        if export_data.ndim == 3 and export_data.shape[2] == 3:
            export_data = cv2.cvtColor(export_data, cv2.COLOR_RGB2BGR)
        save_ext = ext if ext in ('.tif', '.tiff', '.png') else '.tif'
        success, buf = cv2.imencode(save_ext, export_data)

    if success:
        buf.tofile(dest_path)
        return True
    return False


def format_time_seconds(sec: float) -> str:
    """Форматирует секунды в строку MM:SS или HH:MM:SS."""
    if sec < 0 or np.isnan(sec):
        return "—"
    sec_int = int(round(sec))
    m, s = divmod(sec_int, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# =============================================================================
# ФОНОВЫЙ ПОТОК ОБРАБОТКИ ПАКЕТА (QTHREAD WORKER)
# =============================================================================

class BatchWorker(QThread):
    """Фоновый воркер выполнения пакетного конвейера обработки кадров."""

    progress_updated = pyqtSignal(int, int, str, int, int, float, float)
    # (current_index, total, current_filename, width, height, frame_time_ms, eta_seconds)

    frame_completed = pyqtSignal(int, int, str)
    # (current_index, total, saved_file_path)

    batch_finished = pyqtSignal(int, int, list, str)
    # (successful_count, total, failed_files_list, output_directory)

    batch_cancelled = pyqtSignal(int, int, str)
    # (processed_count, total, output_directory)

    def __init__(self, file_paths: list[str], rack_state: dict, batch_options: dict, parent=None):
        super().__init__(parent)
        self.file_paths = list(file_paths)
        self.rack_state = rack_state.copy()
        self.batch_options = batch_options.copy()
        self._abort_requested = False

    def abort(self):
        """Запрос безопасного прерывания процесса."""
        self._abort_requested = True

    def run(self):
        total = len(self.file_paths)
        out_dir = self.batch_options.get('output_dir', '')
        if total == 0:
            self.batch_finished.emit(0, 0, [], out_dir)
            return

        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            self.batch_finished.emit(0, total, [(f, str(e)) for f in self.file_paths], out_dir)
            return

        prefix = self.batch_options.get('prefix', '')
        suffix = self.batch_options.get('suffix', '_rack')
        format_choice = self.batch_options.get('format', '16-bit TIFF (Uncompressed)')
        auto_crop = self.batch_options.get('auto_crop_roi', False)
        auto_rgb_per_frame = self.batch_options.get('auto_rgb_per_frame', True)
        ref_shifts = self.batch_options.get('fixed_shifts', (0.0, 0.0, 0.0, 0.0))
        ref_crop_rect = self.batch_options.get('reference_crop_rect', None)

        successful = 0
        failed = []
        start_total = time.perf_counter()

        for idx, file_path in enumerate(self.file_paths, 1):
            if self._abort_requested:
                self.batch_cancelled.emit(successful, total, out_dir)
                return

            t_start = time.perf_counter()
            base_name = os.path.basename(file_path)
            stem, ext = os.path.splitext(base_name)

            try:
                # 1. Чтение кадра
                raw_frame, orig_dtype, _ = read_image_file(file_path)

                if self._abort_requested:
                    self.batch_cancelled.emit(successful, total, out_dir)
                    return

                # 2. Обработка кадра
                pipeline_opts = {
                    'auto_crop_roi': auto_crop,
                    'reference_crop_rect': ref_crop_rect,
                    'auto_rgb_per_frame': auto_rgb_per_frame,
                    'fixed_shifts': ref_shifts,
                }
                processed = process_frame_pipeline(
                    raw_frame,
                    self.rack_state,
                    options=pipeline_opts,
                    cancel_check=lambda: self._abort_requested
                )

                if self._abort_requested or processed is None:
                    self.batch_cancelled.emit(successful, total, out_dir)
                    return

                # 3. Формирование имени и сохранение
                if "16-bit" in format_choice:
                    out_ext = ".tif"
                elif "8-bit" in format_choice:
                    out_ext = ".png"
                else:
                    out_ext = ext if ext.lower() in ('.tif', '.tiff', '.png') else '.tif'

                out_filename = f"{prefix}{stem}{suffix}{out_ext}"
                dest_path = os.path.join(out_dir, out_filename)

                save_ok = save_processed_frame(dest_path, processed, format_choice, orig_dtype)
                if not save_ok:
                    raise IOError(f"Не удалось записать файл: {dest_path}")

                successful += 1
                t_frame_ms = (time.perf_counter() - t_start) * 1000.0

                elapsed_total = time.perf_counter() - start_total
                avg_time_sec = elapsed_total / successful
                rem_frames = total - idx
                eta_sec = rem_frames * avg_time_sec

                proc_h, proc_w = processed.shape[:2]
                self.progress_updated.emit(idx, total, base_name, proc_w, proc_h, t_frame_ms, eta_sec)
                self.frame_completed.emit(idx, total, dest_path)

            except Exception as e:
                failed.append((file_path, str(e)))
                t_frame_ms = (time.perf_counter() - t_start) * 1000.0
                self.progress_updated.emit(idx, total, f"{base_name} [Ошибка]", 0, 0, t_frame_ms, 0.0)

        self.batch_finished.emit(successful, total, failed, out_dir)


# =============================================================================
# ДИАЛОГ НАСТРОЙКИ ПАКЕТНОЙ ОБРАБОТКИ (BATCH SETUP DIALOG)
# =============================================================================

class BatchSetupDialog(QDialog):
    """Модальное окно настройки пакетной обработки кадров в единой стилистике DAW."""

    SUPPORTED_EXTS = ('.tif', '.tiff', '.png', '.fits', '.fit')

    def __init__(
        self,
        parent=None,
        rack_state: dict = None,
        default_dir: str = "",
        fixed_shifts: tuple = (0.0, 0.0, 0.0, 0.0),
        reference_crop_rect: tuple = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing Setup — AstroRack DAW")
        self.setFixedSize(680, 680)

        self.rack_state = rack_state or {}
        self.default_dir = default_dir or os.path.expanduser("~")
        self.fixed_shifts = fixed_shifts
        self.reference_crop_rect = reference_crop_rect

        self._selected_files = []

        icon_path = get_resource_path(os.path.join("assets", "icons", "icon_32x32.png"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._apply_dark_style()
        self._init_ui()
        self._update_naming_preview()

    def _apply_dark_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #141720;
                color: #f1f5f9;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                background: transparent;
                color: #cbd5e1;
                font-size: 11px;
            }
            QLabel#sectionHeader {
                color: #00adb5;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QFrame.groupCard {
                background-color: #1a1e29;
                border: 1px solid #232a3b;
                border-radius: 6px;
            }
            QLineEdit, QComboBox {
                background-color: #0f121a;
                border: 1px solid #283348;
                border-radius: 4px;
                color: #f1f5f9;
                padding: 4px 8px;
                font-size: 11px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #00adb5;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #161b26;
                border: 1px solid #283348;
                selection-background-color: #00adb5;
                selection-color: #0d1117;
                color: #f1f5f9;
            }
            QListWidget {
                background-color: #0f121a;
                border: 1px solid #283348;
                border-radius: 4px;
                color: #94a3b8;
                font-size: 11px;
            }
            QListWidget::item:selected {
                background-color: #1e293b;
                color: #00fff2;
            }
            QRadioButton, QCheckBox {
                background: transparent;
                color: #e2e8f0;
                font-size: 11px;
                spacing: 6px;
            }
            QRadioButton::indicator, QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
            QCheckBox::indicator:checked {
                background-color: #00adb5;
                border: 1px solid #00fff2;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #0f121a;
                border: 1px solid #334155;
                border-radius: 3px;
            }
            QRadioButton::indicator:checked {
                background-color: #00adb5;
                border: 2px solid #00fff2;
                border-radius: 7px;
            }
            QRadioButton::indicator:unchecked {
                background-color: #0f121a;
                border: 2px solid #334155;
                border-radius: 7px;
            }
            QPushButton {
                background-color: #1e2433;
                border: 1px solid #334155;
                border-radius: 5px;
                color: #cbd5e1;
                font-size: 11px;
                font-weight: 600;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background-color: #273044;
                border-color: #00adb5;
                color: #ffffff;
            }
            QPushButton#btnPrimary {
                background-color: #00adb5;
                border: 1px solid #00fff2;
                color: #0a0d14;
                font-weight: 700;
                font-size: 12px;
            }
            QPushButton#btnPrimary:hover {
                background-color: #00fff2;
                border-color: #ffffff;
            }
            QPushButton#btnPrimary:disabled {
                background-color: #1e293b;
                border-color: #334155;
                color: #64748b;
            }
        """)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        # 0. Заголовок диалога
        header_row = QHBoxLayout()
        header_row.setSpacing(10)
        lbl_icon = QLabel("☷")
        lbl_icon.setStyleSheet("font-size: 22px; color: #00fff2; font-weight: bold;")
        header_row.addWidget(lbl_icon)

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(2)
        lbl_title = QLabel("Пакетная обработка серии кадров (Batch Processing)")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: 800; color: #f1f5f9; letter-spacing: 0.3px;")
        lbl_sub = QLabel("Применение текущего профиля рэка к последовательности астрономических снимков")
        lbl_sub.setStyleSheet("font-size: 10px; color: #94a3b8;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_sub)
        header_row.addLayout(title_vbox)
        header_row.addStretch()
        layout.addLayout(header_row)

        # =====================================================================
        # СЕКЦИЯ А: ИСТОЧНИКИ (SOURCE SELECTION)
        # =====================================================================
        card_src = QFrame()
        card_src.setProperty("class", "groupCard")
        v_src = QVBoxLayout(card_src)
        v_src.setContentsMargins(12, 10, 12, 10)
        v_src.setSpacing(8)

        lbl_sec_a = QLabel("А. ИСТОЧНИКИ (SOURCE SELECTION)")
        lbl_sec_a.setObjectName("sectionHeader")
        v_src.addWidget(lbl_sec_a)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_select_files = QPushButton("📁 Выбрать файлы (Select Files)...")
        self.btn_select_files.clicked.connect(self._on_select_files)
        btn_row.addWidget(self.btn_select_files)

        self.btn_select_folder = QPushButton("📂 Выбрать папку (Select Folder)...")
        self.btn_select_folder.clicked.connect(self._on_select_folder)
        btn_row.addWidget(self.btn_select_folder)

        self.btn_clear_files = QPushButton("✕ Очистить")
        self.btn_clear_files.setFixedWidth(85)
        self.btn_clear_files.clicked.connect(self._on_clear_files)
        btn_row.addWidget(self.btn_clear_files)
        v_src.addLayout(btn_row)

        self.list_files = QListWidget()
        self.list_files.setFixedHeight(110)
        self.list_files.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        v_src.addWidget(self.list_files)

        self.lbl_file_count = QLabel("Найдено файлов: 0 (поддерживаются .tif, .tiff, .png, .fits)")
        self.lbl_file_count.setStyleSheet("color: #00adb5; font-weight: bold;")
        v_src.addWidget(self.lbl_file_count)
        layout.addWidget(card_src)

        # =====================================================================
        # СЕКЦИЯ Б: НАЗНАЧЕНИЕ И ИМЕНОВАНИЕ (DESTINATION & NAMING)
        # =====================================================================
        card_dest = QFrame()
        card_dest.setProperty("class", "groupCard")
        v_dest = QVBoxLayout(card_dest)
        v_dest.setContentsMargins(12, 10, 12, 10)
        v_dest.setSpacing(8)

        lbl_sec_b = QLabel("Б. НАЗНАЧЕНИЕ И ИМЕНОВАНИЕ (DESTINATION & NAMING)")
        lbl_sec_b.setObjectName("sectionHeader")
        v_dest.addWidget(lbl_sec_b)

        # Целевая папка (Радио-группа)
        self.dest_group = QButtonGroup(self)
        self.radio_subfolder = QRadioButton("Создать подпапку внутри исходной:")
        self.radio_subfolder.setChecked(True)
        self.dest_group.addButton(self.radio_subfolder)

        self.edit_subfolder_name = QLineEdit("AstroRack_Batch")
        self.edit_subfolder_name.setFixedWidth(180)
        self.edit_subfolder_name.textChanged.connect(self._update_naming_preview)

        row_sub = QHBoxLayout()
        row_sub.addWidget(self.radio_subfolder)
        row_sub.addWidget(self.edit_subfolder_name)
        row_sub.addStretch()
        v_dest.addLayout(row_sub)

        self.radio_custom_folder = QRadioButton("Выбрать произвольную папку назначения:")
        self.dest_group.addButton(self.radio_custom_folder)

        self.edit_custom_folder = QLineEdit()
        self.edit_custom_folder.setPlaceholderText("Путь к целевой папке...")
        self.edit_custom_folder.setEnabled(False)
        self.edit_custom_folder.textChanged.connect(self._update_naming_preview)

        self.btn_browse_dest = QPushButton("Browse...")
        self.btn_browse_dest.setEnabled(False)
        self.btn_browse_dest.setFixedWidth(80)
        self.btn_browse_dest.clicked.connect(self._on_browse_dest)

        row_custom = QHBoxLayout()
        row_custom.addWidget(self.radio_custom_folder)
        row_custom.addWidget(self.edit_custom_folder, stretch=1)
        row_custom.addWidget(self.btn_browse_dest)
        v_dest.addLayout(row_custom)

        self.radio_subfolder.toggled.connect(self._on_dest_mode_changed)

        # Правило именования
        naming_hbox = QHBoxLayout()
        naming_hbox.setSpacing(10)

        naming_hbox.addWidget(QLabel("Префикс:"))
        self.edit_prefix = QLineEdit("")
        self.edit_prefix.setPlaceholderText("(нет)")
        self.edit_prefix.setFixedWidth(80)
        self.edit_prefix.textChanged.connect(self._update_naming_preview)
        naming_hbox.addWidget(self.edit_prefix)

        naming_hbox.addWidget(QLabel("Суффикс:"))
        self.edit_suffix = QLineEdit("_rack")
        self.edit_suffix.setFixedWidth(90)
        self.edit_suffix.textChanged.connect(self._update_naming_preview)
        naming_hbox.addWidget(self.edit_suffix)

        naming_hbox.addWidget(QLabel("Формат:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems([
            "16-bit TIFF (Uncompressed)",
            "8-bit PNG",
            "Исходный формат (Same as Source)"
        ])
        self.combo_format.currentIndexChanged.connect(self._update_naming_preview)
        naming_hbox.addWidget(self.combo_format, stretch=1)
        v_dest.addLayout(naming_hbox)

        self.lbl_naming_preview = QLabel("Пример результата: jupiter_001_rack.tif")
        self.lbl_naming_preview.setStyleSheet("color: #94a3b8; font-style: italic;")
        v_dest.addWidget(self.lbl_naming_preview)
        layout.addWidget(card_dest)

        # =====================================================================
        # СЕКЦИЯ В: ОПЦИИ ОБРАБОТКИ (PROCESSING OPTIONS)
        # =====================================================================
        card_opts = QFrame()
        card_opts.setProperty("class", "groupCard")
        v_opts = QVBoxLayout(card_opts)
        v_opts.setContentsMargins(12, 10, 12, 10)
        v_opts.setSpacing(8)

        lbl_sec_c = QLabel("В. ОПЦИИ ОБРАБОТКИ (PROCESSING OPTIONS)")
        lbl_sec_c.setObjectName("sectionHeader")
        v_opts.addWidget(lbl_sec_c)

        self.chk_auto_crop = QCheckBox("Auto-Crop to Planetary ROI (автоматическое центрирование и кадрирование планеты)")
        self.chk_auto_crop.setChecked(self.reference_crop_rect is not None)
        v_opts.addWidget(self.chk_auto_crop)

        ins4_align = self.rack_state.get('insert4_enabled', False) and self.rack_state.get('auto_rgb_align', False)
        self.chk_rgb_align = QCheckBox("Auto RGB Phase Align per frame (искать сдвиг дисперсии для каждого кадра индивидуально)")
        self.chk_rgb_align.setChecked(bool(ins4_align))
        v_opts.addWidget(self.chk_rgb_align)

        self.chk_sound_finish = QCheckBox("Воспроизвести звуковой сигнал рэка по завершении")
        self.chk_sound_finish.setChecked(sound_engine.is_enabled())
        v_opts.addWidget(self.chk_sound_finish)
        layout.addWidget(card_opts)

        # =====================================================================
        # СЕКЦИЯ Г: УПРАВЛЕНИЕ (START / CANCEL)
        # =====================================================================
        row_actions = QHBoxLayout()
        row_actions.setSpacing(10)

        self.btn_cancel = QPushButton("Отмена (Cancel)")
        self.btn_cancel.setFixedHeight(36)
        self.btn_cancel.setFixedWidth(120)
        self.btn_cancel.clicked.connect(self.reject)
        row_actions.addWidget(self.btn_cancel)

        row_actions.addStretch()

        self.btn_start = QPushButton("🚀 Start Batch Processing")
        self.btn_start.setObjectName("btnPrimary")
        self.btn_start.setFixedHeight(36)
        self.btn_start.setMinimumWidth(210)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start_clicked)
        row_actions.addWidget(self.btn_start)

        layout.addLayout(row_actions)

    def _on_dest_mode_changed(self, is_subfolder: bool):
        self.edit_subfolder_name.setEnabled(is_subfolder)
        self.edit_custom_folder.setEnabled(not is_subfolder)
        self.btn_browse_dest.setEnabled(not is_subfolder)
        self._update_naming_preview()

    def _on_browse_dest(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку для сохранения результатов", self.default_dir
        )
        if folder:
            self.edit_custom_folder.setText(os.path.normpath(folder))
            self._update_naming_preview()

    def _on_select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите кадры для пакетной обработки",
            self.default_dir,
            "Астрономические кадры (*.tif *.tiff *.png *.fits *.fit);;TIFF (*.tif *.tiff);;PNG (*.png);;FITS (*.fits *.fit);;Все файлы (*.*)"
        )
        if files:
            self._add_files(files)

    def _on_select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку со снимками", self.default_dir
        )
        if folder:
            found = []
            for entry in sorted(os.listdir(folder)):
                full = os.path.join(folder, entry)
                if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in self.SUPPORTED_EXTS:
                    found.append(full)
            if found:
                self._add_files(found)
            else:
                QMessageBox.information(
                    self, "Пакетная обработка",
                    f"В выбранной папке не найдено поддерживаемых снимков ({', '.join(self.SUPPORTED_EXTS)})."
                )

    def _add_files(self, paths: list[str]):
        existing_set = set(os.path.normpath(p) for p in self._selected_files)
        for p in paths:
            norm = os.path.normpath(p)
            if norm not in existing_set:
                self._selected_files.append(norm)
                existing_set.add(norm)

        self._selected_files.sort()
        self._refresh_file_list_ui()

    def _on_clear_files(self):
        self._selected_files.clear()
        self._refresh_file_list_ui()

    def _refresh_file_list_ui(self):
        self.list_files.clear()
        for p in self._selected_files:
            self.list_files.addItem(os.path.basename(p))

        n = len(self._selected_files)
        self.lbl_file_count.setText(f"Найдено файлов: {n}")
        self.btn_start.setEnabled(n > 0)
        if n > 0:
            self.default_dir = os.path.dirname(self._selected_files[0])
        self._update_naming_preview()

    def _update_naming_preview(self):
        sample_name = "jupiter_001"
        if self._selected_files:
            sample_name = os.path.splitext(os.path.basename(self._selected_files[0]))[0]

        pref = self.edit_prefix.text().strip()
        suff = self.edit_suffix.text().strip()
        fmt_idx = self.combo_format.currentIndex()

        ext = ".tif"
        if fmt_idx == 1:
            ext = ".png"
        elif fmt_idx == 2:
            if self._selected_files:
                ext = os.path.splitext(self._selected_files[0])[1].lower()
            else:
                ext = ".tif"

        out_name = f"{pref}{sample_name}{suff}{ext}"

        dest_folder = ""
        if self.radio_subfolder.isChecked():
            sub = self.edit_subfolder_name.text().strip() or "AstroRack_Batch"
            dest_folder = f"[Папка исходника]/{sub}/"
        else:
            cust = self.edit_custom_folder.text().strip()
            dest_folder = f"{cust}/" if cust else "[Выберите папку]/"

        self.lbl_naming_preview.setText(f"Пример сохранения: {dest_folder}{out_name}")

    def _on_start_clicked(self):
        if not self._selected_files:
            QMessageBox.warning(self, "Внимание", "Не выбрано ни одного файла для обработки.")
            return

        if self.radio_custom_folder.isChecked():
            cust = self.edit_custom_folder.text().strip()
            if not cust:
                QMessageBox.warning(self, "Внимание", "Укажите целевую папку сохранения.")
                return

        self.accept()

    def get_configuration(self) -> dict:
        """Возвращает словарь параметров, сформированный в диалоге настройки."""
        if self.radio_subfolder.isChecked():
            sub = self.edit_subfolder_name.text().strip() or "AstroRack_Batch"
            first_dir = os.path.dirname(self._selected_files[0]) if self._selected_files else self.default_dir
            target_dir = os.path.normpath(os.path.join(first_dir, sub))
        else:
            target_dir = os.path.normpath(self.edit_custom_folder.text().strip())

        return {
            'files': list(self._selected_files),
            'output_dir': target_dir,
            'prefix': self.edit_prefix.text().strip(),
            'suffix': self.edit_suffix.text().strip(),
            'format': self.combo_format.currentText(),
            'auto_crop_roi': self.chk_auto_crop.isChecked(),
            'reference_crop_rect': self.reference_crop_rect if self.chk_auto_crop.isChecked() else None,
            'auto_rgb_per_frame': self.chk_rgb_align.isChecked(),
            'fixed_shifts': self.fixed_shifts,
            'play_sound': self.chk_sound_finish.isChecked(),
        }


# =============================================================================
# ОКНО ХОДА ВЫПОЛНЕНИЯ И СТАТУСА (BATCH PROGRESS DIALOG)
# =============================================================================

class BatchProgressDialog(QDialog):
    """Окно прогресса пакетной обработки в отдельном потоке с расчетом ETA и защитой от зависания."""

    def __init__(
        self,
        parent=None,
        file_paths: list[str] = None,
        rack_state: dict = None,
        batch_options: dict = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Processing Batch Sequence — AstroRack DAW")
        self.setFixedSize(540, 310)

        self.file_paths = file_paths or []
        self.rack_state = rack_state or {}
        self.batch_options = batch_options or {}

        self.output_dir = self.batch_options.get('output_dir', '')
        self._is_completed = False

        self._apply_dark_style()
        self._init_ui()
        self._start_worker()

    def _apply_dark_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #141720;
                color: #f1f5f9;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                background: transparent;
                color: #cbd5e1;
                font-size: 11px;
            }
            QProgressBar {
                background-color: #0f121a;
                border: 1px solid #283348;
                border-radius: 6px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #00adb5;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #1e2433;
                border: 1px solid #334155;
                border-radius: 5px;
                color: #cbd5e1;
                font-size: 11px;
                font-weight: 600;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #273044;
                border-color: #00adb5;
                color: #ffffff;
            }
            QPushButton#btnAbort {
                background-color: #2d1822;
                border: 1px solid #7f1d1d;
                color: #f87171;
            }
            QPushButton#btnAbort:hover {
                background-color: #451b27;
                border-color: #ef4444;
                color: #ffffff;
            }
            QPushButton#btnOpenFolder {
                background-color: #162a36;
                border: 1px solid #00adb5;
                color: #00fff2;
                font-weight: 700;
            }
            QPushButton#btnOpenFolder:hover {
                background-color: #00adb5;
                color: #0a0d14;
            }
        """)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        # 1. Заголовок
        row_header = QHBoxLayout()
        self.lbl_header = QLabel("Processing Batch Sequence...")
        self.lbl_header.setStyleSheet("font-size: 15px; font-weight: 800; color: #00fff2;")
        row_header.addWidget(self.lbl_header)
        row_header.addStretch()

        self.lbl_fraction = QLabel(f"Кадр 0 из {len(self.file_paths)}")
        self.lbl_fraction.setStyleSheet("font-size: 12px; font-weight: 700; color: #e2e8f0;")
        row_header.addWidget(self.lbl_fraction)
        layout.addLayout(row_header)

        # 2. Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(24)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 3. Информационные строки
        card_info = QFrame()
        card_info.setStyleSheet("background-color: #1a1e29; border: 1px solid #232a3b; border-radius: 6px; padding: 6px;")
        v_info = QVBoxLayout(card_info)
        v_info.setContentsMargins(8, 6, 8, 6)
        v_info.setSpacing(4)

        self.lbl_current_file = QLabel("Инициализация конвейера обработки...")
        self.lbl_current_file.setStyleSheet("color: #f1f5f9; font-weight: 600;")
        v_info.addWidget(self.lbl_current_file)

        self.lbl_timing = QLabel("Предыдущий кадр: — | Расчетное оставшееся время (ETA): —")
        self.lbl_timing.setStyleSheet("color: #94a3b8;")
        v_info.addWidget(self.lbl_timing)

        self.lbl_target_folder = QLabel(f"Папка: {self.output_dir}")
        self.lbl_target_folder.setStyleSheet("color: #64748b; font-size: 10px;")
        v_info.addWidget(self.lbl_target_folder)

        layout.addWidget(card_info)
        layout.addStretch()

        # 4. Кнопки управления
        self.row_buttons = QHBoxLayout()
        self.row_buttons.setSpacing(10)

        self.btn_open_folder = QPushButton("📂 Открыть папку (Open Output Folder)")
        self.btn_open_folder.setObjectName("btnOpenFolder")
        self.btn_open_folder.setFixedHeight(34)
        self.btn_open_folder.setVisible(False)
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        self.row_buttons.addWidget(self.btn_open_folder)

        self.row_buttons.addStretch()

        self.btn_abort = QPushButton("Abort / Stop")
        self.btn_abort.setObjectName("btnAbort")
        self.btn_abort.setFixedHeight(34)
        self.btn_abort.setFixedWidth(130)
        self.btn_abort.clicked.connect(self._on_abort_clicked)
        self.row_buttons.addWidget(self.btn_abort)

        layout.addLayout(self.row_buttons)

    def _start_worker(self):
        self.worker = BatchWorker(
            file_paths=self.file_paths,
            rack_state=self.rack_state,
            batch_options=self.batch_options,
            parent=self
        )
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.batch_finished.connect(self._on_batch_finished)
        self.worker.batch_cancelled.connect(self._on_batch_cancelled)
        self.worker.start()

    def _on_progress_updated(
        self,
        current_idx: int,
        total: int,
        filename: str,
        w: int,
        h: int,
        frame_time_ms: float,
        eta_sec: float
    ):
        percent = int(round((current_idx / max(1, total)) * 100))
        self.progress_bar.setValue(percent)
        self.lbl_fraction.setText(f"Кадр {current_idx} из {total} ({percent}%)")

        res_str = f" [{w}x{h}]" if w > 0 and h > 0 else ""
        self.lbl_current_file.setText(f"Обработан: {filename}{res_str}")

        eta_str = format_time_seconds(eta_sec)
        self.lbl_timing.setText(f"Предыдущий кадр: {frame_time_ms:.0f} мс | Оставшееся время (ETA): {eta_str}")

    def _on_batch_finished(self, successful: int, total: int, failed: list, out_dir: str):
        self._is_completed = True
        self.output_dir = out_dir

        if not failed:
            self.lbl_header.setText(f"Completed successfully: {successful} / {total}")
            self.lbl_header.setStyleSheet("font-size: 15px; font-weight: 800; color: #10b981;")
            self.progress_bar.setValue(100)
            self.lbl_current_file.setText("Все кадры серии успешно сохранены.")
        else:
            self.lbl_header.setText(f"Completed with errors: {successful} / {total} ({len(failed)} err)")
            self.lbl_header.setStyleSheet("font-size: 15px; font-weight: 800; color: #f59e0b;")
            self.lbl_current_file.setText(f"Не удалось обработать {len(failed)} кадров.")

        self._switch_to_completed_state()

        if self.batch_options.get('play_sound', True) and sound_engine.is_enabled():
            sound_engine.play_shutter()

    def _on_batch_cancelled(self, processed: int, total: int, out_dir: str):
        self._is_completed = True
        self.output_dir = out_dir
        self.lbl_header.setText(f"Cancelled by user: {processed} / {total}")
        self.lbl_header.setStyleSheet("font-size: 15px; font-weight: 800; color: #f87171;")
        self.lbl_current_file.setText("Пакетная обработка прервана пользователем.")
        self._switch_to_completed_state()

    def _switch_to_completed_state(self):
        self.btn_abort.setText("Close")
        self.btn_abort.setObjectName("")
        self.btn_abort.setStyleSheet("""
            QPushButton {
                background-color: #1e2433;
                border: 1px solid #334155;
                color: #cbd5e1;
            }
            QPushButton:hover {
                background-color: #273044;
                color: #ffffff;
                border-color: #00adb5;
            }
        """)
        self.btn_abort.clicked.disconnect()
        self.btn_abort.clicked.connect(self.accept)
        self.btn_open_folder.setVisible(True)

    def _on_abort_clicked(self):
        self.btn_abort.setEnabled(False)
        self.btn_abort.setText("Stopping...")
        self.lbl_current_file.setText("Остановка процесса после завершения текущего кадра...")
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.abort()

    def _on_open_folder(self):
        if not self.output_dir or not os.path.exists(self.output_dir):
            QMessageBox.information(self, "Папка", f"Целевая папка еще не создана:\n{self.output_dir}")
            return

        try:
            if hasattr(os, 'startfile'):
                os.startfile(self.output_dir)
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.output_dir)))
        except Exception:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.output_dir)))

    def closeEvent(self, event):
        """Безопасная обработка закрытия окна во время работы фонового потока."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.abort()
            self.worker.wait(2000)
        event.accept()
