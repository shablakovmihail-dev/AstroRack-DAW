#"""Безопасный ввод-вывод изображений (16-bit TIFF/PNG/JPG/FITS), устойчивый к путям с кириллицей."""
import os
import sys
import cv2
import numpy as np

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


def get_resource_path(relative_path: str = "") -> str:
    """
    Возвращает абсолютный путь к ресурсу, корректно работающий как в среде разработки,
    так и внутри упакованного PyInstaller бинарника (через getattr(sys, '_MEIPASS', os.path.abspath("."))).
    """
    base_dir = getattr(sys, '_MEIPASS', os.path.abspath("."))
    if not hasattr(sys, '_MEIPASS') and not os.path.exists(os.path.join(base_dir, "assets")):
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if os.path.exists(os.path.join(candidate, "assets")):
            base_dir = candidate

    if relative_path:
        return os.path.normpath(os.path.join(base_dir, relative_path))
    return base_dir


# Универсальный алиас функции для безопасного доступа к ресурсам с учетом sys._MEIPASS
resource_path = get_resource_path


def load_image(file_path: str) -> np.ndarray:
    """
    Загружает изображение в формате float32 [0.0, 1.0] в цветовом пространстве RGB или Grayscale.
    Использует чтение через буфер памяти для обхода ограничений OpenCV на Unicode/кириллицу в Windows.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.fit', '.fits']:
        if not HAS_ASTROPY:
            raise ImportError("Для чтения файлов FITS требуется библиотека astropy (pip install astropy).")
        with fits.open(file_path) as hdul:
            data = hdul[0].data if hdul[0].data is not None else hdul[1].data
            data = np.nan_to_num(data.astype(np.float32))
            if data.ndim == 3 and data.shape[0] in [3, 4]:
                data = np.transpose(data, (1, 2, 0))[:, :, :3]
            pmin, pmax = np.min(data), np.max(data)
            return (data - pmin) / max(pmax - pmin, 1e-6)

    # Безопасное чтение через numpy буфер
    buf = np.fromfile(file_path, dtype=np.uint8)
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Не удалось декодировать файл изображения: {file_path}")

    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    if raw.dtype == np.uint16:
        return raw.astype(np.float32) / 65535.0
    elif raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.0
    else:
        # Для исходных float форматов нормализуем в [0, 1]
        raw = raw.astype(np.float32)
        pmin, pmax = np.min(raw), np.max(raw)
        return (raw - pmin) / max(pmax - pmin, 1e-6) if pmax > 1.0 else np.clip(raw, 0.0, 1.0)


def save_image(save_path: str, img_float: np.ndarray) -> bool:
    """
    Сохраняет изображение float32 [0.0, 1.0] в 16-битный TIFF/PNG или 8-битный JPEG.
    Использует cv2.imencode + tofile для безопасной работы с кириллицей.
    """
    ext = os.path.splitext(save_path)[1].lower()
    clamped = np.clip(img_float, 0.0, 1.0)

    if ext in ['.jpg', '.jpeg']:
        export_data = (clamped * 255.0).astype(np.uint8)
        if export_data.ndim == 3:
            export_data = cv2.cvtColor(export_data, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(ext, export_data, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    else:
        export_data = (clamped * 65535.0).astype(np.uint16)
        if export_data.ndim == 3:
            export_data = cv2.cvtColor(export_data, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(ext, export_data)

    if success:
        buf.tofile(save_path)
        return True
    return False
