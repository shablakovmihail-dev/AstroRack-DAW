"""Тесты для State Frame Cache и многоуровневого LRU стадий пайплайна."""
import unittest
import numpy as np
import time
from PyQt6.QtWidgets import QApplication
import sys

app = QApplication.instance() or QApplication(sys.argv)

from core.pipeline import PipelineWorker
from ui.main_window import AstroRackWindow


class TestStateCacheAndLRU(unittest.TestCase):
    def setUp(self):
        self.worker = PipelineWorker()
        self.task_id = self.worker._task_id

    def test_pipeline_worker_multilevel_lru_caching(self):
        img = np.random.uniform(0.1, 0.9, (256, 256, 3)).astype(np.float32)
        self.worker.set_source_image(img)
        task_id = self.worker._task_id

        params_a = {
            'global_bypass': False,
            'insert1_enabled': True,
            'w_mode': 0,
            'layers_config': [{'amount': 1.5, 'denoise': 0.0}] * 3,
            'insert2_enabled': True,
            'rl_sigma': 1.2,
            'rl_iters': 2,
            'rl_anti_ringing': False,
            'rl_wet_mix': 1.0,
            '_preview': False,
        }

        params_b = {
            'global_bypass': False,
            'insert1_enabled': False,
            'insert2_enabled': True,
            'rl_sigma': 1.8,
            'rl_iters': 2,
            'rl_anti_ringing': True,
            'rl_wet_mix': 1.0,
            '_preview': False,
        }

        # 1. Первый расчет State A
        res_a1 = self.worker._execute_pipeline(img, params_a, task_id)
        self.assertIsNotNone(res_a1)

        # 2. Первый расчет State B
        res_b1 = self.worker._execute_pipeline(img, params_b, task_id)
        self.assertIsNotNone(res_b1)

        # Проверяем, что кэши стадий хранят несколько ключей (не затерты)
        self.assertGreaterEqual(len(self.worker._cache_stage2), 2)
        self.assertGreaterEqual(len(self.worker._cache_rl_core), 2)

        # 3. Повторный расчет State A: должен попасть в кэш без повторного пересчета
        t0 = time.perf_counter()
        res_a2 = self.worker._execute_pipeline(img, params_a, task_id)
        t_hit = (time.perf_counter() - t0) * 1000.0

        self.assertIsNotNone(res_a2)
        np.testing.assert_allclose(res_a1, res_a2)
        # Попадание в кэш занимает доли миллисекунды (< 5 мс)
        self.assertLess(t_hit, 5.0)

    def test_main_window_state_frame_cache_instant_switch(self):
        win = AstroRackWindow()
        try:
            img = np.random.uniform(0.1, 0.9, (128, 128, 3)).astype(np.float32)
            win._raw_image = img
            win.canvas.set_raw_image(img)

            # Мокаем параметры для State A и State B
            params_a = {'param': 'A', 'global_bypass': False}
            params_b = {'param': 'B', 'global_bypass': False}

            key_a = win._make_state_cache_key(params_a)
            key_b = win._make_state_cache_key(params_b)

            frame_a = np.zeros((128, 128, 3), dtype=np.float32)
            frame_b = np.ones((128, 128, 3), dtype=np.float32)

            # Имитируем получение готовых кадров от воркера
            win._last_dispatched_params = params_a.copy()
            win._on_frame_rendered(frame_a)

            win._last_dispatched_params = params_b.copy()
            win._on_frame_rendered(frame_b)

            self.assertIn(key_a, win._state_frame_cache)
            self.assertIn(key_b, win._state_frame_cache)

            # Проверяем мгновенное извлечение State A из кэша без отправки в воркер
            request_called = []
            win.worker.request_processing = lambda p: request_called.append(p)

            # Переключение на стейт A
            t0 = time.perf_counter()
            # Подменяем get_values инсертов для воспроизведения params_a
            win._make_state_cache_key = lambda p: key_a
            win._execute_render(preview=False)
            t_switch = (time.perf_counter() - t0) * 1000.0

            # Воркер не должен вызываться
            self.assertEqual(len(request_called), 0)
            # Кадр мгновенно переключен
            self.assertEqual(win._current_processed_image[0, 0, 0], 0.0)
            self.assertLess(t_switch, 10.0)
        finally:
            win.close()

    def test_apply_preset_atomicity(self):
        win = AstroRackWindow()
        try:
            render_calls = []
            win.trigger_render_immediate = lambda: render_calls.append("immediate")

            preset = {
                'insert1_enabled': True,
                'w_mode': 0,
                'layers_config': [{'amount': 1.5, 'denoise': 0.0}] * 3,
                'insert2_enabled': True,
                'rl_sigma': 1.5,
                'rl_iters': 5,
                'rl_wet_mix': 50.0,
            }

            # Применяем пресет
            win._apply_preset_dict(preset)

            # Во время применения не должно быть спама; ровно один вызов trigger_render_immediate в конце
            self.assertEqual(len(render_calls), 1)
            self.assertFalse(win._is_applying_state)
        finally:
            win.close()


if __name__ == '__main__':
    unittest.main()
