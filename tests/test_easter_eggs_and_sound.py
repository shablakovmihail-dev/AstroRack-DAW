"""Тесты для микро-фич и пасхалок:
- Терминатор-сплиттер (Lunar Phase Handle on Split View)
- Тактильный звуковой движок (core/sound.py)
- Тумблер отключения звука (Mute UI Sounds)
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QImage, QPainter, QColor

app = QApplication.instance() or QApplication(sys.argv)

from ui.canvas import AstroCanvas
from ui.main_window import AstroRackWindow
from core.sound import sound_engine


class TestEasterEggsAndSound(unittest.TestCase):

    def setUp(self):
        self.win = AstroRackWindow()

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()

    def test_lunar_phase_split_handle_rendering(self):
        canvas = self.win.canvas
        qimg = QImage(100, 100, QImage.Format.Format_ARGB32_Premultiplied)
        qimg.fill(0)
        p = QPainter(qimg)

        # Проверка рендеринга полумесяца для всех 4 ориентаций (0, 90, 180, 270 град)
        for angle in [0.0, 90.0, 180.0, 270.0]:
            canvas._draw_lunar_phase_handle(p, QPointF(50, 50), angle, QColor("#00fff2"))
        p.end()

        # Проверка переключения фазы серпа в зависимости от направления Before/After
        # 1. Вертикальный сплит: обычный (After справа -> 0°) и инвертированный (After слева -> 180°)
        canvas.set_split_orientation("Vertical")
        canvas.set_split_inverted(False)
        self.assertFalse(canvas._item.split_inverted)

        canvas.set_split_inverted(True)
        self.assertTrue(canvas._item.split_inverted)

        # 2. Горизонтальный сплит: обычный (After снизу -> 90°) и инвертированный (After сверху -> 270°)
        canvas.set_split_orientation("Horizontal")
        self.assertEqual(canvas._item.split_orientation, "Horizontal")

    def test_sound_engine_api(self):
        # Проверяем наличие звуковых файлов
        self.assertTrue(os.path.isfile(sound_engine._relay_path))
        self.assertTrue(os.path.isfile(sound_engine._shutter_path))

        # Проверяем вызовы воспроизведения (не вызывают исключений)
        sound_engine.play_relay_click()
        sound_engine.play_shutter()

        # Проверяем переключение Mute
        orig_state = sound_engine.is_enabled()
        toggled = sound_engine.toggle_enabled()
        self.assertEqual(toggled, not orig_state)
        self.assertEqual(sound_engine.is_enabled(), not orig_state)

        # Возвращаем исходное состояние
        sound_engine.set_enabled(orig_state)
        self.assertEqual(sound_engine.is_enabled(), orig_state)

    def test_mute_ui_button_toggle(self):
        btn = self.win.btn_sound
        self.assertIsNotNone(btn)
        self.assertEqual(btn.width(), 18)
        self.assertEqual(btn.height(), 18)

        # Состояние включен
        sound_engine.set_enabled(True)
        self.win._update_sound_button_ui(True)
        self.assertEqual(btn.text(), "🔊")
        self.assertIn("ВКЛ", btn.toolTip())

        # Клик по кнопке (переключение в Mute)
        btn.click()
        self.assertFalse(sound_engine.is_enabled())
        self.assertEqual(btn.text(), "🔇")
        self.assertIn("ВЫКЛ", btn.toolTip())

        # Клик обратно (включение звука)
        btn.click()
        self.assertTrue(sound_engine.is_enabled())
        self.assertFalse(self.win.sound_muted)
        self.assertEqual(btn.text(), "🔊")

    def test_split_view_detailed_tooltip(self):
        btn = self.win.btn_split_view
        self.assertIsNotNone(btn)
        tt = btn.toolTip()
        self.assertIn("Раздельный просмотр (Split View) [S]", tt)
        self.assertIn("ЛКМ + перетаскивание линии", tt)
        self.assertIn("Before ↔ After", tt)
        self.assertIn("Фаза Луны на маркере указывает активную сторону After", tt)
