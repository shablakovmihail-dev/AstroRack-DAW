"""Тесты для сплэш-скрина, интерактивного пустого холста (Zero State) и демо-библиотеки астроснимков."""
import os
import sys

# Добавляем корень проекта в путь поиска модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint, QMimeData, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

# Гарантируем запуск безголового QApplication
app = QApplication.instance() or QApplication(sys.argv)

from ui.splash import AstroSplashScreen
from ui.canvas import AstroCanvas, ZeroStateWidget, ensure_samples_dir
from ui.main_window import AstroRackWindow
from ui.widgets import THEMES


class TestSplashAndWelcomeCanvas(unittest.TestCase):

    def test_splash_screen_structure_and_progress(self):
        splash = AstroSplashScreen()
        self.assertEqual(splash.width(), 540)
        self.assertEqual(splash.height(), 370)
        self.assertEqual(splash.progress_bar.value(), 0)

        splash.set_progress(42, "Loading test module...")
        self.assertEqual(splash.progress_bar.value(), 42)
        self.assertIn("42%", splash.lbl_percent.text())
        self.assertEqual(splash.lbl_status.text(), "Loading test module...")
        splash.close()

    def test_samples_directory_and_demo_menu(self):
        samples_dir = ensure_samples_dir()
        self.assertTrue(os.path.isdir(samples_dir))

        widget = ZeroStateWidget()
        menu = widget._build_samples_menu()
        self.assertIsNotNone(menu)
        actions = menu.actions()
        self.assertGreater(len(actions), 0)

        # Проверяем, что хотя бы один экшен содержит звездочку или имя файла
        action_texts = [a.text() for a in actions]
        has_demo = any("Jupiter" in t or "Moon" in t or "Orion" in t or "★" in t for t in action_texts)
        self.assertTrue(has_demo, f"Demo actions: {action_texts}")
        widget.deleteLater()

    def test_canvas_zero_state_initial_and_hide_on_load(self):
        canvas = AstroCanvas()
        canvas.show()
        QApplication.processEvents()
        self.assertTrue(hasattr(canvas, 'welcome_widget'))
        self.assertTrue(canvas.welcome_widget.isVisible())
        self.assertFalse(canvas.status_overlay.isVisible())

        # Проверяем надписи и иконку в ZeroStateWidget
        self.assertFalse(canvas.welcome_widget.lbl_icon.pixmap().isNull())
        self.assertIn("Перетащите снимок сюда", canvas.welcome_widget.findChildren(type(canvas.welcome_widget.lbl_icon))[1].text())

        # Симулируем загрузку демо-снимка
        samples_dir = ensure_samples_dir()
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
        self.assertGreater(len(files), 0)
        sample_path = os.path.join(samples_dir, files[0])

        received_paths = []
        canvas.fileDropped.connect(received_paths.append)

        # Проверяем сигнал выбора из welcome_widget
        canvas.welcome_widget.fileSelected.emit(sample_path)
        self.assertEqual(received_paths, [sample_path])

        # Загрузка изображения в canvas скрывает welcome_widget
        import numpy as np
        dummy_img = np.zeros((100, 100, 3), dtype=np.float32)
        canvas.set_raw_image(dummy_img)

        self.assertFalse(canvas.welcome_widget.isVisible())
        self.assertTrue(canvas.status_overlay.isVisible())

        # Проверяем show_welcome_state
        canvas.show_welcome_state()
        self.assertTrue(canvas.welcome_widget.isVisible())
        self.assertFalse(canvas.status_overlay.isVisible())

        canvas.deleteLater()

    def test_main_window_open_image_and_samples_integration(self):
        win = AstroRackWindow()
        win.show()
        QApplication.processEvents()

        # Изначально холст в режиме Zero State
        self.assertIsNone(win._raw_image)
        self.assertTrue(win.canvas.welcome_widget.isVisible())

        # Загружаем снимок через open_image
        samples_dir = ensure_samples_dir()
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
        self.assertGreater(len(files), 0)
        sample_path = os.path.join(samples_dir, files[0])

        win.open_image(sample_path)
        QApplication.processEvents()

        self.assertIsNotNone(win._raw_image)
        self.assertFalse(win.canvas.welcome_widget.isVisible())
        self.assertIn(os.path.splitext(files[0])[0], win.lbl_status.text())

        # Проверяем переключение цветовой темы на Welcome Widget
        win._switch_ab_state('B')
        self.assertEqual(win._active_ab, 'B')

        win.close()
        win.deleteLater()

    def test_drag_and_drop_on_canvas(self):
        canvas = AstroCanvas()
        canvas.show()
        QApplication.processEvents()

        samples_dir = ensure_samples_dir()
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
        sample_path = os.path.join(samples_dir, files[0])

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(sample_path)])

        from PyQt6.QtCore import QPointF
        # Проверка dragEnter
        drag_enter = QDragEnterEvent(
            QPoint(100, 100),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        canvas.dragEnterEvent(drag_enter)
        self.assertTrue(drag_enter.isAccepted())
        self.assertTrue(canvas.welcome_widget._is_drag_active)

        # Проверка dragLeave
        from PyQt6.QtGui import QDragLeaveEvent
        drag_leave = QDragLeaveEvent()
        canvas.dragLeaveEvent(drag_leave)
        self.assertFalse(canvas.welcome_widget._is_drag_active)

        # Проверка dropEvent
        received = []
        canvas.fileDropped.connect(received.append)

        drop = QDropEvent(
            QPointF(100.0, 100.0),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        canvas.dropEvent(drop)
        self.assertTrue(drop.isAccepted())
        self.assertEqual(len(received), 1)
        self.assertEqual(os.path.normpath(received[0]), os.path.normpath(sample_path))
        self.assertFalse(canvas.welcome_widget._is_drag_active)

        canvas.close()
        canvas.deleteLater()


if __name__ == "__main__":
    unittest.main()
