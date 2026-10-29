"""Тесты модуля пакетной обработки кадров (Batch Processing Pipeline)."""
import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QEventLoop, QTimer

from ui.batch_processor import (
    detect_planetary_roi,
    apply_fixed_rgb_shifts,
    process_frame_pipeline,
    read_image_file,
    save_processed_frame,
    format_time_seconds,
    BatchWorker,
    BatchSetupDialog
)


# Инициализация QApplication для тестов PyQt
app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestBatchProcessingPipeline(unittest.TestCase):
    """Тестирование изолированной математической логики обработки и ввода-вывода."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="astrorack_batch_test_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_detect_planetary_roi_disk(self):
        """Проверка детекции диска планеты на темном космическом фоне."""
        img = np.zeros((200, 200, 3), dtype=np.float32)
        # Рисуем яркий диск планеты радиусом 30 в центре (100, 100)
        cv2.circle(img, (100, 100), 30, (0.8, 0.7, 0.6), -1)

        x1, y1, x2, y2 = detect_planetary_roi(img, margin_ratio=0.15)
        self.assertLessEqual(x1, 70)
        self.assertGreaterEqual(x2, 130)
        self.assertLessEqual(y1, 70)
        self.assertGreaterEqual(y2, 130)
        self.assertGreater(x2 - x1, 50)
        self.assertGreater(y2 - y1, 50)

    def test_detect_planetary_roi_fallback(self):
        """Проверка безопасного возврата полного кадра при отсутствии выраженного объекта."""
        img = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        x1, y1, x2, y2 = detect_planetary_roi(img)
        self.assertEqual((x1, y1, x2, y2), (0, 0, 100, 100))

    def test_process_frame_pipeline_uint16_preservation(self):
        """Проверка корректной нормализации и сохранения разрядности для uint16 кадров."""
        raw_u16 = np.full((64, 64, 3), 32768, dtype=np.uint16)
        rack_state = {
            'global_bypass': False,
            'insert1_enabled': False,
            'insert2_enabled': False,
            'insert3_enabled': True,
            'tc_black': 0.0,
            'tc_gamma': 1.0,
            'tc_contrast': 1.0,
        }
        res = process_frame_pipeline(raw_u16, rack_state)
        self.assertIsNotNone(res)
        self.assertEqual(res.dtype, np.float32)
        self.assertTrue(np.all(res >= 0.0) and np.all(res <= 1.0))
        # Проверяем, что исходная средняя серая точка ~0.5 сохранена
        self.assertAlmostEqual(float(np.mean(res)), 32768.0 / 65535.0, places=2)

    def test_process_frame_pipeline_all_inserts(self):
        """Проверка работы полного конвейера со всеми активными инсертами."""
        np.random.seed(42)
        raw = np.random.uniform(0.1, 0.9, (80, 80, 3)).astype(np.float32)

        rack_state = {
            'global_bypass': False,
            'insert4_enabled': True,
            'auto_rgb_align': True,
            'extract_green': False,
            'insert1_enabled': True,
            'w_mode': 0,
            'w_l1': 1.5,
            'w_l2': 1.0,
            'w_denoise': 0.02,
            'w_wet_mix': 1.0,
            'insert2_enabled': True,
            'rl_sigma': 1.2,
            'rl_iters': 2,
            'rl_anti_ringing': True,
            'rl_wet_mix': 0.8,
            'insert3_enabled': True,
            'tc_black': 0.05,
            'tc_gamma': 1.1,
            'tc_contrast': 1.2,
            'insert5_enabled': True,
            'lc_method': 'clahe',
            'lc_amount': 1.5,
            'lc_radius': 16.0,
            'lc_wet_mix': 0.7,
            'insert6_enabled': True,
            'cs_saturation': 1.2,
            'cs_mineral': 0.2,
            'cs_bg_neutralize': False,
            'cs_wet_mix': 1.0,
        }

        res = process_frame_pipeline(raw, rack_state, options={'auto_rgb_per_frame': True})
        self.assertIsNotNone(res)
        self.assertEqual(res.shape, (80, 80, 3))
        self.assertEqual(res.dtype, np.float32)
        self.assertTrue(np.all(res >= 0.0) and np.all(res <= 1.0))

    def test_process_frame_pipeline_cancel_check(self):
        """Проверка раннего прерывания расчета при вызове cancel_check."""
        raw = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        rack_state = {'global_bypass': False, 'insert2_enabled': True, 'rl_iters': 10}

        # Отмена сразу
        res = process_frame_pipeline(raw, rack_state, cancel_check=lambda: True)
        self.assertIsNone(res)

    def test_process_frame_pipeline_global_bypass(self):
        """Проверка работы Master Bypass: мгновенный возврат исходного кадра без искажений."""
        raw = np.random.uniform(0.1, 0.9, (50, 50, 3)).astype(np.float32)
        rack_state = {
            'global_bypass': True,
            'insert1_enabled': True,
            'w_l1': 5.0,
            'insert3_enabled': True,
            'tc_contrast': 3.0
        }
        res = process_frame_pipeline(raw, rack_state)
        np.testing.assert_array_almost_equal(res, raw, decimal=5)

    def test_save_and_read_16bit_tiff_cyrillic_path(self):
        """Проверка сохранения и чтения 16-битного TIFF по путям с кириллицей."""
        cyrillic_sub = os.path.join(self.temp_dir, "АстроФото_Тест")
        os.makedirs(cyrillic_sub, exist_ok=True)
        file_path = os.path.join(cyrillic_sub, "планета_юпитер_16бит.tif")

        # Создаем градиент uint16 [0..65535]
        test_data = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
        test_rgb = np.stack([test_data, test_data * 0.5, test_data * 0.8], axis=-1)

        ok = save_processed_frame(file_path, test_rgb, format_choice="16-bit TIFF (Uncompressed)", orig_dtype=np.uint16)
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(file_path))

        # Читаем обратно
        read_back, orig_dtype, is_color = read_image_file(file_path)
        self.assertEqual(orig_dtype, np.uint16)
        self.assertTrue(is_color)
        self.assertEqual(read_back.shape, (64, 64, 3))
        # Проверяем диапазон значений uint16 (> 255)
        self.assertGreater(float(np.max(read_back)), 60000.0)

    def test_save_and_read_8bit_png(self):
        """Проверка сохранения в 8-битный PNG."""
        file_path = os.path.join(self.temp_dir, "test_8bit.png")
        test_rgb = np.ones((32, 32, 3), dtype=np.float32) * 0.75

        ok = save_processed_frame(file_path, test_rgb, format_choice="8-bit PNG", orig_dtype=np.uint8)
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(file_path))

        read_back, orig_dtype, is_color = read_image_file(file_path)
        self.assertEqual(orig_dtype, np.uint8)
        self.assertTrue(is_color)
        self.assertAlmostEqual(float(read_back[0, 0, 0]), 191, delta=2)

    def test_format_time_seconds(self):
        """Проверка форматирования оставшегося времени ETA."""
        self.assertEqual(format_time_seconds(0), "00:00")
        self.assertEqual(format_time_seconds(65), "01:05")
        self.assertEqual(format_time_seconds(3665), "01:01:05")
        self.assertEqual(format_time_seconds(-5), "—")


class TestBatchWorkerExecution(unittest.TestCase):
    """Интеграционное тестирование фонового воркера BatchWorker."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="astrorack_worker_test_")
        self.src_dir = os.path.join(self.temp_dir, "source")
        self.out_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

        # Генерируем 3 тестовых кадра 16-бит
        self.test_files = []
        for i in range(1, 4):
            path = os.path.join(self.src_dir, f"frame_{i:03d}.tif")
            img = np.full((40, 40, 3), i * 15000, dtype=np.uint16)
            cv2.imencode('.tif', img)[1].tofile(path)
            self.test_files.append(path)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_batch_worker_success(self):
        """Проверка успешной пакетной обработки серии кадров."""
        rack_state = {
            'global_bypass': False,
            'insert3_enabled': True,
            'tc_black': 0.0,
            'tc_gamma': 1.0,
            'tc_contrast': 1.0,
        }
        batch_options = {
            'output_dir': self.out_dir,
            'prefix': 'proc_',
            'suffix': '_rack',
            'format': '16-bit TIFF (Uncompressed)',
            'auto_crop_roi': False,
            'auto_rgb_per_frame': False,
            'fixed_shifts': (0.0, 0.0, 0.0, 0.0),
        }

        worker = BatchWorker(self.test_files, rack_state, batch_options)

        progress_calls = []
        finished_results = []
        loop = QEventLoop()

        worker.progress_updated.connect(lambda cur, tot, name, w, h, t, eta: progress_calls.append((cur, tot, name)))
        def on_finished(succ, tot, failed, out):
            finished_results.append((succ, tot, failed, out))
            loop.quit()

        worker.batch_finished.connect(on_finished)
        worker.start()
        loop.exec()

        self.assertEqual(len(finished_results), 1)
        succ, tot, failed, out = finished_results[0]
        self.assertEqual(succ, 3)
        self.assertEqual(tot, 3)
        self.assertEqual(len(failed), 0)
        self.assertEqual(out, self.out_dir)

        # Проверяем имена созданных файлов
        out_files = sorted(os.listdir(self.out_dir))
        expected_files = ["proc_frame_001_rack.tif", "proc_frame_002_rack.tif", "proc_frame_003_rack.tif"]
        self.assertEqual(out_files, expected_files)

        # Проверяем 16-битность сохраненных файлов
        first_out = os.path.join(self.out_dir, out_files[0])
        raw, dtype, _ = read_image_file(first_out)
        self.assertEqual(dtype, np.uint16)

    def test_batch_worker_abort(self):
        """Проверка корректной отмены пакетной обработки по запросу abort()."""
        large_files = []
        for i in range(1, 11):
            path = os.path.join(self.src_dir, f"large_{i:03d}.tif")
            img = np.full((100, 100, 3), 20000, dtype=np.uint16)
            cv2.imencode('.tif', img)[1].tofile(path)
            large_files.append(path)

        rack_state = {
            'global_bypass': False,
            'insert2_enabled': True,
            'rl_sigma': 1.5,
            'rl_iters': 4,
        }
        batch_options = {
            'output_dir': self.out_dir,
            'prefix': '',
            'suffix': '_out',
            'format': '16-bit TIFF (Uncompressed)',
        }

        worker = BatchWorker(large_files, rack_state, batch_options)
        cancelled_results = []
        loop = QEventLoop()

        def on_cancelled(proc, tot, out):
            cancelled_results.append((proc, tot))
            loop.quit()

        worker.batch_cancelled.connect(on_cancelled)
        worker.batch_finished.connect(lambda *args: loop.quit())

        # Используем DirectConnection для мгновенного синхронного вызова abort прямо в потоке воркера
        worker.progress_updated.connect(lambda *args: worker.abort(), Qt.ConnectionType.DirectConnection)

        worker.start()
        loop.exec()

        # Проверяем, что процесс корректно прерван после 1-го кадра и не обработал все 10
        self.assertEqual(len(cancelled_results), 1)
        proc, tot = cancelled_results[0]
        self.assertEqual(proc, 1)
        self.assertEqual(tot, 10)


class TestBatchSetupDialogAndMainWindow(unittest.TestCase):
    """Тестирование диалога BatchSetupDialog и интеграции с MainWindow."""

    def test_setup_dialog_config(self):
        """Проверка сбора конфигурации из диалога BatchSetupDialog."""
        dlg = BatchSetupDialog(
            rack_state={'insert1_enabled': True},
            default_dir="C:/TestDir",
            fixed_shifts=(0.5, -0.2, -0.4, 0.1)
        )
        self.assertIsNotNone(dlg)

        # Добавляем 2 файла
        dlg._add_files(["C:/TestDir/frame1.tif", "C:/TestDir/frame2.tif"])
        self.assertEqual(len(dlg._selected_files), 2)
        self.assertTrue(dlg.btn_start.isEnabled())

        # Настраиваем префикс и суффикс
        dlg.edit_prefix.setText("stack_")
        dlg.edit_suffix.setText("_enhanced")
        dlg.combo_format.setCurrentIndex(0)  # 16-bit TIFF

        cfg = dlg.get_configuration()
        self.assertEqual(len(cfg['files']), 2)
        self.assertEqual(cfg['prefix'], "stack_")
        self.assertEqual(cfg['suffix'], "_enhanced")
        self.assertIn("16-bit", cfg['format'])
        self.assertIn("AstroRack_Batch", cfg['output_dir'])

    def test_main_window_has_batch_buttons(self):
        """Проверка наличия точек входа для Batch Process в AstroRackWindow."""
        from ui.main_window import AstroRackWindow
        win = AstroRackWindow()
        try:
            self.assertTrue(hasattr(win, 'btn_batch'), "Кнопка btn_batch должна присутствовать на top_bar")
            self.assertTrue(hasattr(win, 'batch_btn'), "Алиас batch_btn должен присутствовать")
            self.assertTrue(hasattr(win, '_open_batch_processor'), "Метод _open_batch_processor должен присутствовать")
            self.assertTrue(hasattr(win, 'open_batch_processing'), "Алиас open_batch_processing должен присутствовать")

            # 1. По умолчанию при отсутствии изображения кнопка неактивна
            self.assertFalse(win.btn_batch.isEnabled())
            self.assertIn("Загрузите референсный снимок", win.btn_batch.toolTip())

            # 2. При наличии изображения кнопка активна
            win._raw_image = np.ones((64, 64, 3), dtype=np.float32)
            win._update_canvas_guard_state()
            self.assertTrue(win.btn_batch.isEnabled())
            self.assertIn("Применить текущий профиль рэка", win.btn_batch.toolTip())

            # 3. При сбросе изображения кнопка снова деактивируется
            win._raw_image = None
            win._update_canvas_guard_state()
            self.assertFalse(win.btn_batch.isEnabled())
            self.assertIn("Загрузите референсный снимок", win.btn_batch.toolTip())

            # 4. Проверка guard-условия: вызов без изображения безопасно выходит
            win._open_batch_processor()
            win.open_batch_processing()
        finally:
            win.close()


if __name__ == '__main__':
    unittest.main()
