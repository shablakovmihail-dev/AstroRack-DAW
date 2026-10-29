import unittest
import os
import tempfile
import datetime
from PIL import Image

from core.metadata_reader import (
    parse_date_string,
    extract_capture_date,
    TAG_DATETIME,
    TAG_DATETIME_ORIGINAL
)


class TestMetadataReader(unittest.TestCase):
    def test_parse_date_string_formats(self):
        # 1. EXIF standard
        dt = parse_date_string("2024:05:15 22:30:45")
        self.assertEqual(dt, datetime.datetime(2024, 5, 15, 22, 30, 45))

        # 2. ISO standard
        dt = parse_date_string("2023-08-30T19:20:10Z")
        self.assertEqual(dt, datetime.datetime(2023, 8, 30, 19, 20, 10))

        # 3. Space-separated ISO
        dt = parse_date_string("2023-08-30 19:20:10")
        self.assertEqual(dt, datetime.datetime(2023, 8, 30, 19, 20, 10))

        # 4. Date only
        dt = parse_date_string("2025-12-31")
        self.assertEqual(dt, datetime.datetime(2025, 12, 31, 0, 0, 0))

        # 5. Invalid / corrupt inputs
        self.assertIsNone(parse_date_string(""))
        self.assertIsNone(parse_date_string(None))
        self.assertIsNone(parse_date_string("corrupt date string"))
        self.assertIsNone(parse_date_string("1850-01-01"))  # Year out of range
        self.assertIsNone(parse_date_string("2150-01-01"))

    def test_extract_png_exif(self):
        tmp_path = os.path.join(tempfile.gettempdir(), "test_moon_exif.png")
        try:
            img = Image.new("RGB", (10, 10), color="gray")
            exif = img.getexif()
            exif[TAG_DATETIME_ORIGINAL] = "2024:07:21 21:15:00"
            img.save(tmp_path, exif=exif)
            img.close()

            extracted = extract_capture_date(tmp_path)
            self.assertEqual(extracted, datetime.datetime(2024, 7, 21, 21, 15, 0))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def test_extract_tiff_datetime(self):
        tmp_path = os.path.join(tempfile.gettempdir(), "test_moon_meta.tiff")
        try:
            img = Image.new("L", (10, 10), color=128)
            exif = img.getexif()
            exif[TAG_DATETIME] = "2023:10:28 20:00:00"
            img.save(tmp_path, exif=exif)
            img.close()

            extracted = extract_capture_date(tmp_path)
            self.assertEqual(extracted, datetime.datetime(2023, 10, 28, 20, 0, 0))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def test_extract_fits_header(self):
        tmp_path = os.path.join(tempfile.gettempdir(), "test_moon_fits.fits")
        try:
            # Создаем минимальный валидный FITS блок (2880 байт) с тегом DATE-OBS
            hdr_card = "SIMPLE  =                    T / FITS file                                      BITPIX  =                   16 / 16 bits                                        NAXIS   =                    0 / No image data                                  DATE-OBS= '2023-08-30T21:40:00' / Observation date UTC                         END                                                                             "
            hdr_bytes = hdr_card.ljust(2880, " ").encode("ascii")
            with open(tmp_path, "wb") as f:
                f.write(hdr_bytes)

            extracted = extract_capture_date(tmp_path)
            self.assertEqual(extracted, datetime.datetime(2023, 8, 30, 21, 40, 0))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def test_extract_nonexistent_or_no_exif(self):
        self.assertIsNone(extract_capture_date("nonexistent_image_file.png"))

        tmp_path = os.path.join(tempfile.gettempdir(), "test_no_exif.png")
        try:
            img = Image.new("RGB", (8, 8), color="white")
            img.save(tmp_path)
            img.close()
            self.assertIsNone(extract_capture_date(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
