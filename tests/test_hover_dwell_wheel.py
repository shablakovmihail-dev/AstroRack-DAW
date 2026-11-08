"""Тесты для механизма безопасной прокрутки колесика мыши (Hover Dwell & Focus)."""
import unittest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QWheelEvent, QMouseEvent
import sys

app = QApplication.instance() or QApplication(sys.argv)

from ui.widgets import (
    PlanetarySlider,
    RotaryKnob,
    AstroDoubleSpinBox,
    AstroSpinBox,
    AstroComboBox,
    AstroParamRow,
    WaveletInsert,
    LocalContrastInsert,
)


class TestHoverDwellWheel(unittest.TestCase):
    def _create_wheel_event(self, delta_y=120):
        return QWheelEvent(
            QPointF(5, 5),
            QPointF(5, 5),
            QPoint(0, 0),
            QPoint(0, delta_y),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )

    def test_planetary_slider_dwell_and_focus(self):
        slider = PlanetarySlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)

        # 1. Скролл без фокуса и без задержки курсора — событие должно игнорироваться
        ev = self._create_wheel_event(120)
        slider.wheelEvent(ev)
        self.assertFalse(ev.isAccepted())
        self.assertEqual(slider.value(), 50)

        # 2. Имитация удержания курсора (Hover Dwell 250 мс)
        slider._hover_dwell_timer.timeout.emit()
        self.assertTrue(slider._wheel_enabled)
        ev2 = self._create_wheel_event(120)
        slider.wheelEvent(ev2)
        self.assertTrue(ev2.isAccepted())
        self.assertNotEqual(slider.value(), 50)

        # 3. Курсор ушел (leaveEvent) — колесико снова заблокировано
        slider.leaveEvent(None)
        self.assertFalse(slider._wheel_enabled)
        val_before = slider.value()
        ev3 = self._create_wheel_event(120)
        slider.wheelEvent(ev3)
        self.assertFalse(ev3.isAccepted())
        self.assertEqual(slider.value(), val_before)

    def test_rotary_knob_dwell_and_focus(self):
        knob = RotaryKnob(default_val=50)

        # 1. Без задержки — игнорирование
        ev = self._create_wheel_event(120)
        knob.wheelEvent(ev)
        self.assertFalse(ev.isAccepted())
        self.assertEqual(knob.value(), 50)

        # 2. После таймаута задержки — вращение разрешено
        knob._hover_dwell_timer.timeout.emit()
        ev2 = self._create_wheel_event(120)
        knob.wheelEvent(ev2)
        self.assertTrue(ev2.isAccepted())
        self.assertEqual(knob.value(), 52)

        # 3. После ухода курсора — снова заблокировано
        knob.leaveEvent(None)
        ev3 = self._create_wheel_event(120)
        knob.wheelEvent(ev3)
        self.assertFalse(ev3.isAccepted())
        self.assertEqual(knob.value(), 52)

    def test_astro_spinbox_and_combobox(self):
        dsp = AstroDoubleSpinBox()
        dsp.setRange(0.0, 10.0)
        dsp.setValue(2.0)

        sp = AstroSpinBox()
        sp.setRange(0, 100)
        sp.setValue(10)

        cb = AstroComboBox()
        cb.addItems(["A", "B", "C"])
        cb.setCurrentIndex(0)

        for w in (dsp, sp, cb):
            # Без dwell
            ev = self._create_wheel_event(120)
            w.wheelEvent(ev)
            self.assertFalse(ev.isAccepted(), f"{type(w)} must ignore wheel without dwell/focus")

            # С dwell
            w._hover_dwell_timer.timeout.emit()
            ev2 = self._create_wheel_event(120)
            w.wheelEvent(ev2)
            self.assertTrue(ev2.isAccepted(), f"{type(w)} must accept wheel with dwell")

    def test_astro_param_row_integration(self):
        row = AstroParamRow("Test Param", 0.0, 10.0, 5.0, 0.1, 2)
        self.assertIsInstance(row.slider, PlanetarySlider)
        self.assertIsInstance(row.spinbox, AstroDoubleSpinBox)

        # Проверка, что контролы строки по умолчанию игнорируют колесико
        ev_s = self._create_wheel_event(120)
        row.slider.wheelEvent(ev_s)
        self.assertFalse(ev_s.isAccepted())

        ev_sp = self._create_wheel_event(120)
        row.spinbox.wheelEvent(ev_sp)
        self.assertFalse(ev_sp.isAccepted())

    def test_insert_combo_modes(self):
        ins1 = WaveletInsert()
        ins5 = LocalContrastInsert()
        self.assertIsInstance(ins1.combo_mode, AstroComboBox)
        self.assertIsInstance(ins5.combo_mode, AstroComboBox)


if __name__ == "__main__":
    unittest.main()
