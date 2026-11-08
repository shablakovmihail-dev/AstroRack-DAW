"""Тесты модуля калибровки масштаба линейки по известным кратерам (Crater Scale Calibration)."""

import unittest
import math
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QMouseEvent

# Ensure single QApplication instance
app = QApplication.instance() or QApplication([])

from core.constants import LUNAR_CRATERS, DEFAULT_LUNAR_DISTANCE_KM
from ui.dialogs import CraterCalibrationDialog
from ui.canvas import AstroCanvas
from ui.measurement_hud import MeasurementHUD


class TestCraterScaleCalibration(unittest.TestCase):
    """Набор тестов для базы кратеров, диалога калибровки и интеграции с линейкой."""

    def test_lunar_craters_database(self):
        """Проверка наличия и точности справочных диаметров лунных кратеров."""
        expected_craters = {
            "Tycho": 85.0,
            "Copernicus": 93.0,
            "Plato": 101.0,
            "Clavius": 225.0,
            "Theophilus": 100.0,
            "Aristarchus": 40.0,
            "Kepler": 31.0,
            "Langrenus": 132.0,
            "Gassendi": 110.0,
        }
        for name, dia in expected_craters.items():
            self.assertIn(name, LUNAR_CRATERS)
            self.assertEqual(LUNAR_CRATERS[name], dia)

    def test_crater_calibration_dialog_recalculation(self):
        """Проверка расчета масштаба км/пикс и arcsec/пикс в диалоге калибровки."""
        px_length = 100.0
        dialog = CraterCalibrationDialog(current_px_length=px_length, lunar_distance_km=384400.0)

        # По умолчанию выбран Tycho (85.0 км)
        self.assertEqual(dialog.combo_craters.currentText(), "Tycho")
        self.assertEqual(dialog.spin_diameter.value(), 85.0)

        # scale_km_per_px = 85.0 / 100.0 = 0.85
        self.assertAlmostEqual(dialog.scale_km_per_px, 0.85, places=5)

        # arcsec_per_px = (0.85 / 384400.0) * (180 * 3600 / pi)
        expected_arcsec = (0.85 / 384400.0) * (180.0 * 3600.0 / math.pi)
        self.assertAlmostEqual(dialog.scale_arcsec_per_px, expected_arcsec, places=4)

        # Переключение на Plato (101.0 км)
        dialog.combo_craters.setCurrentText("Plato")
        self.assertEqual(dialog.spin_diameter.value(), 101.0)
        self.assertAlmostEqual(dialog.scale_km_per_px, 1.01, places=5)

        # Переключение на Custom и ввод произвольного значения
        dialog.combo_craters.setCurrentText("Custom (Ручной ввод)")
        dialog.spin_diameter.setValue(50.0)
        self.assertAlmostEqual(dialog.scale_km_per_px, 0.5, places=5)

    def test_ruler_hud_buttons_and_scale_box(self):
        """Проверка доступности кнопки 'Калибровать' и плашки линейки в HUD."""
        hud = MeasurementHUD()
        hud.set_mode("ruler")

        # В режиме ruler плашка линейки видима
        self.assertFalse(hud.ruler_scale_box.isHidden())
        # Пока отрезок не нарисован, кнопка калибровки отключена
        self.assertFalse(hud.btn_calibrate_ruler.isEnabled())
        # Кнопка сброса калибровки скрыта
        self.assertTrue(hud.btn_reset_ruler_scale.isHidden())

        # Передаем результат замера отрезка длиной 150 px
        hud.set_ruler_results(
            projected_km=75.0,
            ang_str='40.2"',
            dist_px=150.0,
            ruler_text='150.0 px | 75.00 km | 40.20"'
        )
        # Кнопка калибровки должна стать активной
        self.assertTrue(hud.btn_calibrate_ruler.isEnabled())
        self.assertIn("150.0 px | 75.00 km", hud.lbl_ruler_scale.text())

        # Устанавливаем статус калибровки
        hud.set_scale_calibrated(True, 0.5)
        self.assertFalse(hud.btn_reset_ruler_scale.isHidden())

        # Сброс калибровки
        hud.set_scale_calibrated(False)
        self.assertTrue(hud.btn_reset_ruler_scale.isHidden())

        # Очистка результатов линейки
        hud.clear_results()
        self.assertFalse(hud.btn_calibrate_ruler.isEnabled())

    def test_canvas_crater_scale_calibration_workflow(self):
        """Проверка полного цикла калибровки линейки на AstroCanvas."""
        canvas = AstroCanvas()
        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("ruler")
        canvas.set_telescope_params(1000.0, 3.76)
        canvas.set_lunar_distance(384400.0)

        # Базовый масштаб оптики: (3.76 / (1000 * 1000)) * 384400 = 1.445344 км/пикс
        default_scale = (3.76 / (1000.0 * 1000.0)) * 384400.0
        self.assertAlmostEqual(canvas.scale_km_per_px, default_scale, places=5)

        # Чертим отрезок линейки от (100, 100) до (200, 100) => 100 px
        canvas._ruler_p1 = QPointF(100.0, 100.0)
        canvas._ruler_p2 = QPointF(200.0, 100.0)
        canvas._emit_ruler_info()

        # Кнопка калибровки в HUD активна
        self.assertTrue(canvas.measurement_hud.btn_calibrate_ruler.isEnabled())
        # Проверяем формат строки в HUD: "{length_px:.1f} px | {length_km:.2f} km | {length_arcsec:.2f}\""
        hud_text = canvas.measurement_hud.lbl_ruler_scale.text()
        self.assertIn("100.0 px", hud_text)
        self.assertIn(f"{100.0 * default_scale:.2f} km", hud_text)

        # Имитируем калибровку по кратеру Tycho (85 км на 100 пикселей => 0.85 км/пикс)
        canvas.scale_km_per_px = 0.85
        canvas.measurement_hud.set_scale_calibrated(True, 0.85)
        canvas._emit_ruler_info()

        self.assertEqual(canvas.scale_km_per_px, 0.85)
        hud_text_calib = canvas.measurement_hud.lbl_ruler_scale.text()
        # Должно быть 100.0 px | 85.00 km | ...
        self.assertIn("100.0 px | 85.00 km", hud_text_calib)
        self.assertTrue(canvas.measurement_hud.btn_reset_ruler_scale.isVisible())

        # Следующее измерение: отрезок длиной 200 px должен дать 170.00 km
        canvas._ruler_p2 = QPointF(300.0, 100.0)
        canvas._emit_ruler_info()
        hud_text_200 = canvas.measurement_hud.lbl_ruler_scale.text()
        self.assertIn("200.0 px | 170.00 km", hud_text_200)

        # Сброс калибровки к значениям по умолчанию
        canvas.reset_scale_calibration()
        self.assertAlmostEqual(canvas.scale_km_per_px, default_scale, places=5)
        self.assertFalse(canvas.measurement_hud.btn_reset_ruler_scale.isVisible())
        hud_text_reset = canvas.measurement_hud.lbl_ruler_scale.text()
        self.assertIn("200.0 px", hud_text_reset)
        self.assertIn(f"{200.0 * default_scale:.2f} km", hud_text_reset)


if __name__ == "__main__":
    unittest.main()
