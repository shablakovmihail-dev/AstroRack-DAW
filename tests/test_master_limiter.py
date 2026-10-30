"""Тесты мастер-лимитера светов (Master Soft-Clip Limiter) и цветовой схемы мастер-панели."""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from core.processing import AstroEngine, soft_clip_limiter
from core.pipeline import PipelineWorker
from ui.widgets import compute_clipping_metrics
from ui.main_window import AstroRackWindow

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class TestMasterSoftClipLimiterMath(unittest.TestCase):
    """Тестирование математики мягкого тангенциального колена компрессии светов."""

    def test_linear_below_threshold(self):
        """Значения ниже порога T=0.88 передаются строго линейно 1:1 (y = x)."""
        x = np.array([0.0, 0.1, 0.5, 0.75, 0.88], dtype=np.float32)
        y = soft_clip_limiter(x, threshold=0.88)
        np.testing.assert_allclose(y, x, atol=1e-6)

    def test_soft_knee_above_threshold(self):
        """Значения выше порога сжимаются по формуле T + (1-T)*tanh((x-T)/(1-T))."""
        T = 0.88
        scale = 1.0 - T
        x = np.array([0.90, 0.95, 1.00], dtype=np.float32)
        expected = T + scale * np.tanh((x - T) / scale)
        y = soft_clip_limiter(x, threshold=T)
        np.testing.assert_allclose(y, expected, atol=1e-6)

    def test_no_hard_clipping_at_one(self):
        """При входном значении 1.0 выход строго меньше 1.0 (ни один пиксель не упирается в 1.0)."""
        x = np.ones((10, 10), dtype=np.float32)
        y = soft_clip_limiter(x, threshold=0.88)
        self.assertTrue(np.all(y < 0.99))
        self.assertAlmostEqual(float(y.max()), 0.88 + 0.12 * np.tanh(1.0), places=5)

    def test_derivative_smoothness_continuity(self):
        """Проверка непрерывности функции и производной в точке порога T."""
        T = 0.88
        eps = 1e-4
        y_minus = soft_clip_limiter(np.array([T - eps], dtype=np.float32), threshold=T)[0]
        y_T = soft_clip_limiter(np.array([T], dtype=np.float32), threshold=T)[0]
        y_plus = soft_clip_limiter(np.array([T + eps], dtype=np.float32), threshold=T)[0]

        # Непрерывность значений
        self.assertAlmostEqual(y_minus, T - eps, places=5)
        self.assertAlmostEqual(y_T, T, places=5)
        self.assertAlmostEqual(y_plus, T + (1.0 - T) * np.tanh(eps / (1.0 - T)), places=5)

        # Непрерывность производной (dy/dx ≈ 1.0 с обеих сторон)
        deriv_left = (y_T - y_minus) / eps
        deriv_right = (y_plus - y_T) / eps
        self.assertAlmostEqual(deriv_left, 1.0, places=3)
        self.assertAlmostEqual(deriv_right, 1.0, places=3)

    def test_multichannel_rgb_support(self):
        """Корректная обработка цветных 3D массивов (H, W, 3)."""
        rgb = np.zeros((4, 4, 3), dtype=np.float32)
        rgb[:, :, 0] = 0.50   # ниже порога
        rgb[:, :, 1] = 0.95   # выше порога
        rgb[:, :, 2] = 1.00   # пересвет
        out = soft_clip_limiter(rgb, threshold=0.88)
        self.assertEqual(out.shape, (4, 4, 3))
        np.testing.assert_allclose(out[:, :, 0], 0.50, atol=1e-6)
        self.assertTrue(np.all(out[:, :, 1] < 0.95))
        self.assertTrue(np.all(out[:, :, 2] < 0.98))

    def test_clipping_metrics_drop_to_zero(self):
        """При активации лимитера выбитые пиксели пересвета падают в 0."""
        # Кадр с пересветом
        blown_img = np.array([
            [0.2, 0.5, 0.88],
            [0.92, 0.999, 1.0]
        ], dtype=np.float32)
        metrics_before = compute_clipping_metrics(blown_img)
        self.assertGreater(metrics_before['highlight_count'], 0)
        self.assertTrue(metrics_before['highlight_clip'])

        # После лимитера
        limited_img = soft_clip_limiter(blown_img, threshold=0.88)
        metrics_after = compute_clipping_metrics(limited_img)
        self.assertEqual(metrics_after['highlight_count'], 0)
        self.assertEqual(metrics_after['highlight_pct'], 0.0)
        self.assertFalse(metrics_after['highlight_clip'])


class TestPipelineLimiterIntegration(unittest.TestCase):
    """Тестирование применения лимитера в цепочке PipelineWorker."""

    def test_pipeline_master_limiter_execution(self):
        worker = PipelineWorker()
        raw_img = np.ones((20, 20), dtype=np.float32)

        # 1. Лимитер выключен
        params_off = {
            'global_bypass': False,
            'master_limiter_enabled': False,
        }
        res_off = worker._execute_pipeline(raw_img, params_off, worker._task_id)
        self.assertIsNotNone(res_off)
        self.assertAlmostEqual(float(res_off.max()), 1.0, places=4)

        # 2. Лимитер включен
        params_on = {
            'global_bypass': False,
            'master_limiter_enabled': True,
            'limiter_threshold': 0.88,
        }
        res_on = worker._execute_pipeline(raw_img, params_on, worker._task_id)
        self.assertIsNotNone(res_on)
        self.assertLess(float(res_on.max()), 0.98)

    def test_pipeline_global_bypass_disables_limiter(self):
        """Если Master Bypass активен, лимитер отключается вместе со всей цепочкой."""
        worker = PipelineWorker()
        raw_img = np.ones((20, 20), dtype=np.float32)

        params_bypass = {
            'global_bypass': True,
            'master_limiter_enabled': True,
        }
        res = worker._execute_pipeline(raw_img, params_bypass, worker._task_id)
        self.assertIs(res, raw_img)
        self.assertEqual(float(res.max()), 1.0)


class TestMasterPanelThemeSync(unittest.TestCase):
    """Тестирование кнопки Master Limiter и связки палитры мастер-секции с активным стейтом [ A ] / [ B ]."""

    def setUp(self):
        self.win = AstroRackWindow()

    def tearDown(self):
        self.win.close()

    def test_master_limiter_button_attributes(self):
        """Наличие кнопки Master Limiter, свойства checkable и начальное состояние."""
        self.assertTrue(hasattr(self.win, 'btn_master_limiter'))
        self.assertTrue(self.win.btn_master_limiter.isCheckable())
        self.assertFalse(self.win.btn_master_limiter.isChecked())
        self.assertEqual(self.win.btn_master_limiter.text(), "LIMITER: OFF")

        # Переключение
        self.win.btn_master_limiter.setChecked(True)
        self.assertEqual(self.win.btn_master_limiter.text(), "LIMITER: ACTIVE")

        self.win.btn_master_limiter.setChecked(False)
        self.assertEqual(self.win.btn_master_limiter.text(), "LIMITER: OFF")

    def test_master_accent_theme_state_a(self):
        """Цветовая схема в State A: циан #00e5ff."""
        self.win.update_master_accent_theme('A')

        # 1. Кнопка стейта
        self.assertIn("State: [ A ]", self.win.btn_state_toggle.text())
        ss_state = self.win.btn_state_toggle.styleSheet()
        self.assertIn("#00e5ff", ss_state)

        # 2. Master Bypass (Effects ON)
        self.win.btn_global_bypass.setChecked(False)
        self.win.update_master_accent_theme('A')
        ss_bypass = self.win.btn_global_bypass.styleSheet()
        self.assertIn("#00e5ff", ss_bypass)

        # 3. Master Limiter (ACTIVE)
        self.win.btn_master_limiter.setChecked(True)
        self.win.update_master_accent_theme('A')
        ss_limiter_active = self.win.btn_master_limiter.styleSheet()
        self.assertIn("#00e5ff", ss_limiter_active)

        # Master Limiter (OFF)
        self.win.btn_master_limiter.setChecked(False)
        self.win.update_master_accent_theme('A')
        ss_limiter_off = self.win.btn_master_limiter.styleSheet()
        self.assertIn("#4a5568", ss_limiter_off)

        # 4. Плавающий бейдж холста
        ss_badge = self.win.canvas.status_overlay.styleSheet()
        self.assertIn("#00e5ff", ss_badge)

    def test_master_accent_theme_state_b(self):
        """Цветовая схема в State B: янтарь #f59e0b."""
        self.win.update_master_accent_theme('B')

        # 1. Кнопка стейта
        self.assertIn("State: [ B ]", self.win.btn_state_toggle.text())
        ss_state = self.win.btn_state_toggle.styleSheet()
        self.assertIn("#f59e0b", ss_state)

        # 2. Master Bypass (Effects ON)
        self.win.btn_global_bypass.setChecked(False)
        self.win.update_master_accent_theme('B')
        ss_bypass = self.win.btn_global_bypass.styleSheet()
        self.assertIn("#f59e0b", ss_bypass)

        # 3. Master Limiter (ACTIVE)
        self.win.btn_master_limiter.setChecked(True)
        self.win.update_master_accent_theme('B')
        ss_limiter_active = self.win.btn_master_limiter.styleSheet()
        self.assertIn("#f59e0b", ss_limiter_active)

        # Master Limiter (OFF)
        self.win.btn_master_limiter.setChecked(False)
        self.win.update_master_accent_theme('B')
        ss_limiter_off = self.win.btn_master_limiter.styleSheet()
        self.assertIn("#4a5568", ss_limiter_off)

        # 4. Плавающий бейдж холста
        ss_badge = self.win.canvas.status_overlay.styleSheet()
        self.assertIn("#f59e0b", ss_badge)

    def test_master_bypass_active_style(self):
        """При активном Master Bypass стиль кнопки приглушенный нейтральный/красноватый."""
        self.win.btn_global_bypass.setChecked(True)
        self.win.update_master_accent_theme('A')
        ss_bypassed = self.win.btn_global_bypass.styleSheet()
        self.assertIn("#c07882", ss_bypassed)

    def test_switch_state_synchronization(self):
        """Переключение _switch_ab_state синхронизирует тему мастера."""
        self.win._switch_ab_state('B')
        self.assertEqual(self.win._active_ab, 'B')
        self.assertIn("#f59e0b", self.win.btn_state_toggle.styleSheet())
        self.assertIn("#f59e0b", self.win.canvas.status_overlay.styleSheet())

        self.win._switch_ab_state('A')
        self.assertEqual(self.win._active_ab, 'A')
        self.assertIn("#00e5ff", self.win.btn_state_toggle.styleSheet())
        self.assertIn("#00e5ff", self.win.canvas.status_overlay.styleSheet())


if __name__ == '__main__':
    unittest.main()
