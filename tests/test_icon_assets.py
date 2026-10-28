"""Тесты для пакета иконок, скрипта генерации и интеграции в UI AstroRack DAW."""
import os
import sys

# Добавляем корень проекта в путь поиска модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from PIL import Image
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

app = QApplication.instance() or QApplication(sys.argv)

from ui.splash import AstroSplashScreen
from ui.dialogs import AboutDialog
from ui.main_window import AstroRackWindow


class TestIconAssetsAndIntegration(unittest.TestCase):

    def test_icons_generation_files_and_transparency(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icons_dir = os.path.join(root, "assets", "icons")
        ico_path = os.path.join(root, "assets", "app_icon.ico")

        expected_sizes = [16, 24, 32, 48, 64, 128, 256, 512]
        for size in expected_sizes:
            p = os.path.join(icons_dir, f"icon_{size}x{size}.png")
            self.assertTrue(os.path.isfile(p), f"Missing icon: {p}")

            with Image.open(p) as img:
                self.assertEqual(img.size, (size, size))
                self.assertEqual(img.mode, "RGBA")
                # Угол должен быть прозрачным (alpha == 0)
                corner_alpha = img.getpixel((0, 0))[3]
                self.assertEqual(corner_alpha, 0, f"Corner not transparent at {p}")

        # Проверяем ICO файл
        self.assertTrue(os.path.isfile(ico_path), f"Missing ICO: {ico_path}")
        with Image.open(ico_path) as ico:
            sizes = ico.info.get("sizes", set())
            for s in [16, 24, 32, 48, 64, 128, 256]:
                self.assertIn((s, s), sizes)

    def test_splash_screen_logo_integration(self):
        splash = AstroSplashScreen()
        self.assertTrue(hasattr(splash, "lbl_center_logo"))
        self.assertFalse(splash.lbl_center_logo.pixmap().isNull())
        self.assertEqual(splash.lbl_center_logo.width(), 128)
        self.assertEqual(splash.lbl_center_logo.height(), 128)
        self.assertFalse(splash.windowIcon().isNull())
        splash.deleteLater()

    def test_about_dialog_logo_integration(self):
        dlg = AboutDialog()
        self.assertFalse(dlg.windowIcon().isNull())
        self.assertTrue(hasattr(dlg, "lbl_icon"))
        self.assertFalse(dlg.lbl_icon.pixmap().isNull())
        self.assertEqual(dlg.lbl_icon.width(), 64)
        self.assertEqual(dlg.lbl_icon.height(), 64)
        dlg.deleteLater()

    def test_main_window_logo_and_window_icon(self):
        win = AstroRackWindow()
        self.assertFalse(win.windowIcon().isNull())
        self.assertTrue(hasattr(win, "lbl_logo"))
        self.assertFalse(win.lbl_logo.pixmap().isNull())
        win.close()
        win.deleteLater()

    def test_welcome_canvas_hat_logo_integration(self):
        from ui.canvas import AstroCanvas
        canvas = AstroCanvas()
        self.assertTrue(hasattr(canvas.welcome_widget, "lbl_icon"))
        self.assertFalse(canvas.welcome_widget.lbl_icon.pixmap().isNull())
        self.assertEqual(canvas.welcome_widget.lbl_icon.width(), 96)
        self.assertEqual(canvas.welcome_widget.lbl_icon.height(), 96)
        canvas.deleteLater()


if __name__ == "__main__":
    unittest.main()
