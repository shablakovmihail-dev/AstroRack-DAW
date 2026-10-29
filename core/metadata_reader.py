"""
Модуль извлечения метаданных и даты съёмки астрономических снимков.
Поддерживает форматы TIFF, PNG, JPEG, а также FITS (DATE-OBS / DATE).
"""

import os
import re
import datetime
from typing import Optional
from PIL import Image

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


# Стандартные EXIF теги для даты и времени
TAG_DATETIME = 306            # 0x0132
TAG_DATETIME_ORIGINAL = 36867 # 0x9003
TAG_DATETIME_DIGITIZED = 36868 # 0x9004
IFD_EXIF = 0x8769


def parse_date_string(date_str: str) -> Optional[datetime.datetime]:
    """
    Парсит строковое представление даты и времени в объект datetime.datetime (UTC).
    Поддерживает стандартные форматы EXIF, ISO 8601, FITS DATE-OBS.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    cleaned = date_str.strip().strip("'\"")
    if not cleaned:
        return None

    # Попытка парсинга через datetime.fromisoformat (Python 3.11+)
    try:
        # Заменяем разделители двоеточия в дате "YYYY:MM:DD" на дефисы для ISO
        iso_candidate = cleaned
        if re.match(r"^\d{4}:\d{2}:\d{2}", cleaned):
            iso_candidate = cleaned[:10].replace(":", "-") + cleaned[10:]
        
        # Обработка суффикса Z
        if iso_candidate.endswith("Z"):
            iso_candidate = iso_candidate[:-1] + "+00:00"

        dt = datetime.datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        if 1900 <= dt.year <= 2100:
            return dt
    except Exception:
        pass

    # Набор явных паттернов
    patterns = [
        # EXIF standard: 2024:05:15 22:30:00
        "%Y:%m:%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S.%f",
        # ISO standard: 2024-05-15 22:30:00 / 2024-05-15T22:30:00
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        # Slash standard: 2024/05/15 22:30:00
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        # Date only
        "%Y:%m:%d",
        "%Y-%m-%d",
        "%Y/%m/%d",
        # FITS old style: 15/05/24 or 15/05/2024
        "%d/%m/%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y",
        "%d-%m-%Y %H:%M:%S",
    ]

    for pat in patterns:
        try:
            dt = datetime.datetime.strptime(cleaned, pat)
            if 1900 <= dt.year <= 2100:
                return dt
        except ValueError:
            continue

    return None


def _extract_from_fits(file_path: str) -> Optional[datetime.datetime]:
    """Извлечение даты из FITS-заголовка через astropy или прямой парсинг FITS Header."""
    # 1. Попытка через astropy.io.fits
    if HAS_ASTROPY:
        try:
            with fits.open(file_path) as hdul:
                for hdu in hdul:
                    hdr = getattr(hdu, "header", None)
                    if hdr is not None:
                        for key in ("DATE-OBS", "DATE", "DATE_OBS"):
                            val = hdr.get(key)
                            if val:
                                dt = parse_date_string(str(val))
                                if dt is not None:
                                    return dt
        except Exception:
            pass

    # 2. Быстрый fallback через сканирование заголовков FITS (ASCII карточки по 80 байт)
    try:
        with open(file_path, "rb") as f:
            # Читаем до 10 блоков по 2880 байт
            header_bytes = f.read(2880 * 10)
            header_str = header_bytes.decode("ascii", errors="ignore")
            # Ищем DATE-OBS = '...' или DATE = '...'
            match = re.search(r"DATE-OBS\s*=\s*'([^']+)'", header_str)
            if not match:
                match = re.search(r"DATE\s*=\s*'([^']+)'", header_str)
            if match:
                return parse_date_string(match.group(1))
    except Exception:
        pass

    return None


def extract_capture_date(file_path: str) -> Optional[datetime.datetime]:
    """
    Пытается извлечь дату и время съёмки из метаданных изображения.

    Поддерживаемые форматы:
    - TIFF / PNG / JPEG: теги EXIF (DateTimeOriginal, DateTimeDigitized, DateTime) и info.
    - FITS: теги DATE-OBS или DATE.

    Возвращает:
        datetime.datetime в UTC или None, если метаданные отсутствуют или не удалось распарсить.
    """
    if not file_path or not os.path.isfile(file_path):
        return None

    ext = os.path.splitext(file_path)[1].lower()

    # FITS форматы
    if ext in (".fit", ".fits", ".fts"):
        return _extract_from_fits(file_path)

    # Стандартные растровые форматы через Pillow
    try:
        with Image.open(file_path) as img:
            # 1. Проверяем EXIF
            exif = img.getexif()
            if exif:
                # Проверяем IFD_EXIF (0x8769)
                try:
                    ifd_exif = exif.get_ifd(IFD_EXIF)
                except Exception:
                    ifd_exif = {}

                # Порядок приоритета тегов: DateTimeOriginal -> DateTimeDigitized -> DateTime
                for tag_id in (TAG_DATETIME_ORIGINAL, TAG_DATETIME_DIGITIZED, TAG_DATETIME):
                    val = ifd_exif.get(tag_id) or exif.get(tag_id)
                    if val:
                        dt = parse_date_string(str(val))
                        if dt is not None:
                            return dt

            # 2. Проверяем атрибут img.info (актуально для PNG tEXt/zTXt, TIFF тегов)
            info = getattr(img, "info", {})
            if isinstance(info, dict):
                for key in (
                    "DateTimeOriginal",
                    "DateTime",
                    "date:create",
                    "date:modify",
                    "Creation Time",
                    "date-obs",
                    "DATE-OBS",
                    "date",
                ):
                    val = info.get(key)
                    if val:
                        dt = parse_date_string(str(val))
                        if dt is not None:
                            return dt

    except Exception:
        pass

    return None
