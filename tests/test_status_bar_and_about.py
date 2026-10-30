"""Тесты для нижнего статус-бара с телеметрией и диалогового окна 'О программе'."""
import os
import sys

# Добавляем корень проекта в путь поиска модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtCore import Qt, QPointF

app = QApplication.instance() or QApplication(sys.argv)

from ui.dialogs import AboutDialog
from ui.canvas import AstroCanvas
from ui.main_window import AstroRackWindow


class TestStatusBarAndAbout(unittest.TestCase):

    def test_about_dialog_content_and_manifest(self):
        dlg = AboutDialog()
        self.assertEqual(dlg.width(), 620)
        self.assertEqual(dlg.height(), 680)

        # Проверяем тексты внутри окна
        labels = [lbl.text() for lbl in dlg.findChildren(QLabel)]
        has_title = any("AstroRack DAW" in t for t in labels)
        has_version = any("v8.3.6" in t for t in labels)
        has_license = any("Public Domain" in t or "Zero Rights Reserved" in t for t in labels)
        has_pyqt = any("PyQt6" in t for t in labels)
        has_opencv = any("OpenCV" in t for t in labels)
        has_astana = any("Astana, KZ" in t for t in labels)
        has_epoch = any("Epoch J2026.10" in t for t in labels)

        self.assertTrue(has_title, "AstroRack DAW title missing")
        self.assertTrue(has_version, "v8.3.6 badge missing")
        self.assertTrue(has_license, "Public domain manifesto missing")
        self.assertTrue(has_pyqt, "PyQt6 tech badge missing")
        self.assertTrue(has_opencv, "OpenCV tech badge missing")
        self.assertTrue(has_astana, "Astana Observatory badge missing")
        self.assertTrue(has_epoch, "Epoch J2026.10 telemetry missing")
        dlg.deleteLater()

    def test_canvas_live_pixel_probe_and_zoom(self):
        canvas = AstroCanvas()
        canvas.show()
        QApplication.processEvents()

        # Изначально без изображения
        probes = []
        canvas.pixelProbeChanged.connect(lambda c, v: probes.append((c, v)))

        canvas._emit_pixel_probe(QPointF(50.0, 50.0))
        self.assertEqual(probes[-1], ("X: ---, Y: ---", ""))

        # Загружаем цветное float-изображение 200x100
        test_img = np.zeros((100, 200, 3), dtype=np.float32)
        test_img[40, 50] = [0.12, 0.45, 0.88]
        canvas.set_raw_image(test_img)

        # Проверяем зонд внутри изображения
        canvas._emit_pixel_probe(QPointF(50.0, 40.0))
        coords, val = probes[-1]
        self.assertEqual(coords, "X: 50, Y: 40")
        self.assertIn("RGB: [0.12, 0.45, 0.88]", val)

        # Проверяем зонд вне изображения
        canvas._emit_pixel_probe(QPointF(250.0, 40.0))
        self.assertEqual(probes[-1], ("X: ---, Y: ---", ""))

        # Проверяем монохромное изображение
        mono_img = np.ones((50, 50), dtype=np.float32) * 0.75
        canvas.set_processed_image(mono_img)
        canvas._emit_pixel_probe(QPointF(10.0, 10.0))
        coords, val = probes[-1]
        self.assertEqual(coords, "X: 10, Y: 10")
        self.assertIn("L: 0.750", val)

        # Проверяем зум
        self.assertIn("Zoom:", canvas.get_current_zoom_str())
        canvas.deleteLater()

    def test_main_window_status_bar_metrics(self):
        win = AstroRackWindow()
        win.show()
        QApplication.processEvents()

        # Проверяем заголовок окна строго по спецификации релиза
        self.assertEqual(win.windowTitle(), "AstroRack DAW v8.3.6 — Lunar & Planetary Processor")

        # Проверяем наличие секций статус-бара
        self.assertTrue(hasattr(win, 'lbl_sb_pipeline'))
        self.assertTrue(hasattr(win, 'lbl_sb_specs'))
        self.assertTrue(hasattr(win, 'lbl_sb_zoom'))
        self.assertTrue(hasattr(win, 'lbl_sb_coords'))
        self.assertTrue(hasattr(win, 'lbl_sb_probe'))

        self.assertEqual(win.lbl_sb_pipeline.text(), "Ready")
        self.assertEqual(win.lbl_sb_specs.text(), "No Image Loaded")
        self.assertIn("Zoom: 100%", win.lbl_sb_zoom.text())
        self.assertEqual(win.lbl_sb_coords.text(), "X: ---, Y: ---")

        # Проверяем наличие кнопки [ ? ] в верхнем тулбаре
        self.assertTrue(hasattr(win, 'btn_about'))
        self.assertEqual(win.btn_about.text(), "?")
        self.assertEqual(win.btn_about.width(), 36)
        self.assertEqual(win.btn_about.height(), 36)

        # Загружаем тестовый снимок в главное окно
        from ui.canvas import ensure_samples_dir
        samples_dir = ensure_samples_dir()
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
        self.assertGreater(len(files), 0)
        sample_path = os.path.join(samples_dir, files[0])

        win.open_image(sample_path)
        QApplication.processEvents()

        # Проверяем обновление параметров снимка
        self.assertNotEqual(win.lbl_sb_specs.text(), "No Image Loaded")
        self.assertIn("px |", win.lbl_sb_specs.text())

        # Симулируем событие движения мыши на холсте
        win.canvas._emit_pixel_probe(QPointF(10.0, 10.0))
        self.assertEqual(win.lbl_sb_coords.text(), "X: 10, Y: 10")
        self.assertTrue(len(win.lbl_sb_probe.text()) > 0)

        win.close()
        win.deleteLater()


if __name__ == "__main__":
    unittest.main()
