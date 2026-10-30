"""Тесты спектро-яркостного анализатора гистограммы:
- Логарифмический (LOG) и линейный (LIN) рендеринг
- Мультиканальный анализ LUM (Rec.709) vs RGB
- Аппаратные светодиодные индикаторы клиппинга (Shadows & Highlights)
- Интерактивные визиры Black Point и White Point
- Шапка переключателей и кнопка справки [?]
- Окно контекстной справки HistogramHelpDialog
- Empty Canvas Guard и Master Bypass
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent
from PyQt6.QtGui import QPainter, QImage

app = QApplication.instance() or QApplication(sys.argv)

from ui.widgets import AstroHistogramWidget, AstroHistogramHeader, THEMES
from ui.dialogs import HistogramHelpDialog
from ui.main_window import AstroRackWindow
from ui.canvas import ensure_samples_dir


class TestHistogramAnalyzer(unittest.TestCase):

    def setUp(self):
        self.widget = AstroHistogramWidget()
        self.header = AstroHistogramHeader()
        self.header.attach_histogram(self.widget)

    def test_log_vs_lin_calculation(self):
        # Тестовые данные с ярким пиком в нуле (черный космос) и небольшим полезным сигналом
        data = np.zeros((100, 100), dtype=np.float32)
        data[40:60, 40:60] = 0.5  # полезный сигнал

        self.widget.update_histogram(data)
        self.assertTrue(self.widget._has_data)

        # Режим LOG (по умолчанию)
        self.widget.set_scale_mode("LOG")
        log_hist = self.widget._hist_lum.copy()

        # Режим LIN
        self.widget.set_scale_mode("LIN")
        lin_hist = self.widget._hist_lum.copy()

        # При LOG значении полутона должны быть значительно выше, чем при LIN из-за сжатия экстремального спайка в 0
        bin_idx = int(0.5 * 255.0)  # bin 127
        self.assertGreater(log_hist[bin_idx], lin_hist[bin_idx])

    def test_lum_vs_rgb_mode(self):
        # Создаем цветное изображение (100x100x3)
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[:, :, 0] = 0.8  # Red
        img[:, :, 1] = 0.4  # Green
        img[:, :, 2] = 0.1  # Blue

        self.widget.update_histogram(img)
        self.assertTrue(self.widget._is_rgb_image)

        # LUM по Rec.709: Y = 0.2126*0.8 + 0.7152*0.4 + 0.0722*0.1 ≈ 0.463
        expected_y_bin = int((0.2126 * 0.8 + 0.7152 * 0.4 + 0.0722 * 0.1) * 255.0)
        self.assertEqual(np.argmax(self.widget._raw_hist_lum), expected_y_bin)

        # Проверяем пики в индивидуальных каналах R, G, B
        self.assertEqual(np.argmax(self.widget._raw_hist_r), int(0.8 * 255.0))
        self.assertEqual(np.argmax(self.widget._raw_hist_g), int(0.4 * 255.0))
        self.assertEqual(np.argmax(self.widget._raw_hist_b), int(0.1 * 255.0))

        # Переключение режимов через шапку
        self.header.btn_rgb.click()
        self.assertEqual(self.widget._channel_mode, "RGB")
        self.header.btn_lum.click()
        self.assertEqual(self.widget._channel_mode, "LUM")

    def test_clipping_indicators(self):
        # 1. Тест Shadow Clipping (> 0.05% чистых нулей)
        img_shadow = np.full((100, 100), 0.5, dtype=np.float32)
        # Добавляем 10 нулей (0.1% кадра > 0.05%)
        img_shadow.ravel()[:10] = 0.0
        self.widget.update_histogram(img_shadow)
        self.assertTrue(self.widget._shadow_clip)
        self.assertFalse(self.widget._highlight_clip)

        # 2. Тест Highlight Clipping (наличие пикселей со значением 1.0 / 255)
        img_hl = np.full((100, 100), 0.5, dtype=np.float32)
        img_hl[0, 0] = 1.0  # пересвет
        self.widget.update_histogram(img_hl)
        self.assertFalse(self.widget._shadow_clip)
        self.assertTrue(self.widget._highlight_clip)

    def test_visors_synchronization(self):
        # Устанавливаем визиры отсечения диапазона
        self.widget.set_cut_points(black_val=0.150, gamma_val=1.50, black_on=True, gamma_on=True)
        self.assertTrue(self.widget._black_point_enabled)
        self.assertAlmostEqual(self.widget._black_point_norm, 0.150, places=3)
        self.assertTrue(self.widget._white_point_enabled)
        # При gamma=1.50 white_point_norm = 1.0 / 1.50 ≈ 0.667
        self.assertAlmostEqual(self.widget._white_point_norm, 1.0 / 1.50, places=2)

    def test_theme_and_bypass_sync(self):
        # State A (Cyan #00E5FF)
        self.widget.set_state_ab("A")
        self.assertEqual(self.widget._theme_accent, "#00E5FF")

        # State B (Amber #FF9E0B)
        self.widget.set_state_ab("B")
        self.assertEqual(self.widget._theme_accent, "#FF9E0B")

        # Master Bypass
        self.widget.set_master_bypass(True)
        self.assertTrue(self.widget._is_master_bypassed)
        self.widget.set_master_bypass(False)
        self.assertFalse(self.widget._is_master_bypassed)

    def test_empty_canvas_and_help_button_guard(self):
        # При пустом холсте (set_canvas_enabled(False))
        self.header.set_canvas_enabled(False)
        self.assertFalse(self.header.btn_log.isEnabled())
        self.assertFalse(self.header.btn_lin.isEnabled())
        self.assertFalse(self.header.btn_lum.isEnabled())
        self.assertFalse(self.header.btn_rgb.isEnabled())
        # Кнопка [ ? ] справки ДОЛЖНА ОСТАВАТЬСЯ активной!
        self.assertTrue(self.header.btn_help.isEnabled())

        # При загрузке кадра (set_canvas_enabled(True))
        self.header.set_canvas_enabled(True)
        self.assertTrue(self.header.btn_log.isEnabled())
        self.assertTrue(self.header.btn_lin.isEnabled())
        self.assertTrue(self.header.btn_lum.isEnabled())
        self.assertTrue(self.header.btn_rgb.isEnabled())
        self.assertTrue(self.header.btn_help.isEnabled())

    def test_help_dialog(self):
        dlg = HistogramHelpDialog()
        self.assertIsNotNone(dlg)
        self.assertEqual(dlg.windowTitle(), "Справка: Анализ гистограммы — AstroRack DAW")
        # Проверяем наличие кнопки закрытия
        self.assertIsNotNone(dlg.btn_ok)
        dlg.close()

    def test_roi_mode_auto_vs_full(self):
        # 1. Создаем кадр 120x120: черный космос (0.0) и яркий диск планеты (0.75) в центре
        img = np.zeros((120, 120), dtype=np.float32)
        # Рисуем круг радиусом 25 пикселей в центре (60, 60)
        yy, xx = np.ogrid[:120, :120]
        circle_mask = ((xx - 60) ** 2 + (yy - 60) ** 2) <= 25 ** 2
        img[circle_mask] = 0.75

        # По умолчанию режим FULL
        self.widget.update_histogram(img)
        self.assertEqual(self.widget.get_roi_mode(), "FULL")
        self.assertTrue(self.widget._shadow_clip)  # Черный космос доминирует
        full_bin0_count = self.widget._raw_hist_lum[0]
        self.assertGreater(full_bin0_count, 10000)

        # Переключаем в AUTO (Auto-Mask ROI)
        self.header.btn_auto.click()
        self.assertEqual(self.widget.get_roi_mode(), "AUTO")
        auto_bin0_count = self.widget._raw_hist_lum[0]

        # В режиме AUTO космос отсечен: в нулевом бине нет пикселей диска!
        self.assertEqual(auto_bin0_count, 0)
        self.assertFalse(self.widget._shadow_clip)
        # Основной пик диска на 0.75 * 255 ≈ 191
        peak_bin = int(np.argmax(self.widget._raw_hist_lum))
        self.assertEqual(peak_bin, int(0.75 * 255.0))

        # Переключаем обратно в FULL: проверяем мгновенное восстановление из кэша
        self.header.btn_full.click()
        self.assertEqual(self.widget.get_roi_mode(), "FULL")
        self.assertEqual(self.widget._raw_hist_lum[0], full_bin0_count)

        # 2. Проверка fallback при отсутствии объекта (равномерный шум или пустота)
        blank = np.zeros((50, 50), dtype=np.float32)
        self.widget.set_roi_mode("AUTO")
        self.widget.update_histogram(blank)
        # Fallback к полному кадру без исключений
        self.assertTrue(self.widget._has_data)
        self.assertEqual(self.widget._raw_hist_lum[0], 2500)

    def test_roi_header_buttons_and_theme_sync(self):
        # Проверка блокировки кнопок FULL/AUTO при пустом холсте
        self.header.set_canvas_enabled(False)
        self.assertFalse(self.header.btn_full.isEnabled())
        self.assertFalse(self.header.btn_auto.isEnabled())

        # Разблокировка
        self.header.set_canvas_enabled(True)
        self.assertTrue(self.header.btn_full.isEnabled())
        self.assertTrue(self.header.btn_auto.isEnabled())

        # Синхронизация темы A/B
        self.header.apply_theme({'name': 'A', 'bright': '#00fff2'})
        self.assertEqual(self.header._accent, '#00E5FF')

        self.header.apply_theme({'name': 'B', 'bright': '#fbbf24'})
        self.assertEqual(self.header._accent, '#FF9E0B')

    def test_crop_overlay_diagonal_and_center_reticle(self):
        from ui.canvas import CropOverlayItem
        crop = CropOverlayItem(400, 300)
        self.assertEqual(crop.img_w, 400)
        self.assertEqual(crop.img_h, 300)

        # Проверка рендеринга без ошибок на QImage
        qimg = QImage(400, 300, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(qimg)
        crop.paint(painter, None)
        painter.end()

        # Центр рамки кропа
        r = crop.crop_rect
        center = r.center()
        self.assertAlmostEqual(center.x(), 200.0, places=1)
        self.assertAlmostEqual(center.y(), 150.0, places=1)

    def test_exact_clipping_metrics_and_formatting(self):
        from ui.widgets import format_clipping_text, compute_clipping_metrics

        # 1. Форматирование текста
        self.assertEqual(format_clipping_text(0, 0.0), "0% (0 px)")
        self.assertEqual(format_clipping_text(5, 0.005), "< 0.01% (5 px)")
        self.assertEqual(format_clipping_text(1234, 0.1234), "0.123% (1,234 px)")

        # 2. Точный расчёт выбитых пикселей на кадрах
        # 200x200 = 40,000 пикселей. Ровно 3 пикселя в нуле (< 0.01%) и 80 пикселей в пересвете (0.200%)
        test_frame = np.full((200, 200), 0.5, dtype=np.float32)
        test_frame[0, 0:3] = 0.0
        test_frame[10, :80] = 1.0

        metrics = compute_clipping_metrics(test_frame)
        self.assertEqual(metrics['shadow_count'], 3)
        self.assertAlmostEqual(metrics['shadow_pct'], (3 / 40000.0) * 100.0, places=4)
        self.assertTrue(metrics['shadow_clip'])

        self.assertEqual(metrics['highlight_count'], 80)
        self.assertAlmostEqual(metrics['highlight_pct'], (80 / 40000.0) * 100.0, places=4)
        self.assertTrue(metrics['highlight_clip'])

        # Обновление в виджете
        self.widget.update_histogram(test_frame)
        self.assertEqual(self.widget.get_shadow_count(), 3)
        self.assertEqual(self.widget.get_shadow_text(), "< 0.01% (3 px)")
        self.assertEqual(self.widget.get_highlight_count(), 80)
        self.assertEqual(self.widget.get_highlight_text(), f"{(80/40000.0)*100.0:.3f}% (80 px)")

    def test_clipping_indicator_interactive_clicks_and_signals(self):
        from PyQt6.QtGui import QMouseEvent
        # Устанавливаем изображение
        img = np.full((50, 50), 0.5, dtype=np.float32)
        img[0, 0] = 0.0
        img[0, 1] = 1.0
        self.widget.update_histogram(img)

        received_modes = []
        self.widget.clipping_overlay_requested.connect(received_modes.append)

        # Клик по левому индикатору (тени)
        shadow_rect = self.widget.get_shadow_rect()
        click_pos = shadow_rect.center().toPoint()
        press_event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(click_pos),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        self.widget.mousePressEvent(press_event)
        self.assertIn("shadows", received_modes)

        # Синхронизация состояния Checked
        self.widget.set_clipping_overlay_mode("shadows")
        self.assertEqual(self.widget.get_clipping_overlay_mode(), "shadows")

        # Клик по правому индикатору (светы)
        hl_rect = self.widget.get_highlight_rect()
        click_pos_hl = hl_rect.center().toPoint()
        press_event_hl = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(click_pos_hl),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        self.widget.mousePressEvent(press_event_hl)
        self.assertIn("highlights", received_modes)

        # Проверка рендеринга виджета в активном состоянии (с темной полупрозрачной подложкой)
        self.widget.set_clipping_overlay_mode("shadows")
        qimg = QImage(240, 84, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(qimg)
        self.widget.paintEvent(None)
        painter.end()

        # Сброс
        self.widget.set_clipping_overlay_mode(None)
        self.assertIsNone(self.widget.get_clipping_overlay_mode())

    def test_canvas_sonar_overlay_and_radar(self):
        from ui.canvas import AstroCanvas
        from PyQt6.QtGui import QKeyEvent
        canvas = AstroCanvas()

        # Создаем изображение 100x100: черные пиксели (0.0) и пересвеченные (1.0)
        img = np.full((100, 100), 0.4, dtype=np.float32)
        img[10:15, 10:15] = 0.0  # Кластер теней 5x5
        img[80:85, 80:85] = 1.0  # Кластер светов 5x5

        canvas.set_processed_image(img)

        # 1. Активация режима теней ("shadows")
        canvas.set_clipping_overlay_mode("shadows")
        self.assertEqual(canvas.clipping_overlay_mode, "shadows")
        self.assertIsNotNone(canvas._clipping_mask_pixmap)
        self.assertGreater(len(canvas._clipping_centroids), 0)
        self.assertTrue(canvas._radar_timer.isActive())

        # Проверка центроида
        c0 = canvas._clipping_centroids[0]
        self.assertAlmostEqual(c0.x(), 12.0, delta=2.0)
        self.assertAlmostEqual(c0.y(), 12.0, delta=2.0)

        # 2. Активация режима светов ("highlights")
        canvas.set_clipping_overlay_mode("highlights")
        self.assertEqual(canvas.clipping_overlay_mode, "highlights")
        self.assertIsNotNone(canvas._clipping_mask_pixmap)
        self.assertGreater(len(canvas._clipping_centroids), 0)
        c_hl = canvas._clipping_centroids[0]
        self.assertAlmostEqual(c_hl.x(), 82.0, delta=2.0)
        self.assertAlmostEqual(c_hl.y(), 82.0, delta=2.0)

        # 3. Переключение (toggle) того же режима отключает подсветку
        canvas.toggle_clipping_overlay("highlights")
        self.assertIsNone(canvas.clipping_overlay_mode)
        self.assertIsNone(canvas._clipping_mask_pixmap)
        self.assertFalse(canvas._radar_timer.isActive())

        # 4. Нажатие Escape отключает подсветку
        canvas.set_clipping_overlay_mode("shadows")
        self.assertEqual(canvas.clipping_overlay_mode, "shadows")
        esc_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(esc_event)
        self.assertIsNone(canvas.clipping_overlay_mode)
        self.assertFalse(canvas._radar_timer.isActive())

    def test_tooltips_and_help_text(self):
        # 1. Тултип шапки [?]
        help_tip = self.header.btn_help.toolTip()
        self.assertIn("Гистограмма распределения яркости (Luminance Histogram):", help_tip)
        self.assertIn("радарными кругами", help_tip)

        # 2. Тултип теней
        shadow_rect = self.widget.get_shadow_rect()
        pos_s = shadow_rect.center().toPoint()
        self.assertTrue(self.widget._is_in_shadow_zone(pos_s))

        # 3. Тултип светов
        hl_rect = self.widget.get_highlight_rect()
        pos_h = hl_rect.center().toPoint()
        self.assertTrue(self.widget._is_in_highlight_zone(pos_h))

    def test_safe_overlay_generation_and_no_crash(self):
        from ui.canvas import AstroCanvas
        from PyQt6.QtGui import QPainter
        from PyQt6.QtCore import QRectF

        canvas = AstroCanvas()

        # 1. Тест с различными типами данных и размерностями
        # 8-bit RGB
        img_u8 = np.zeros((64, 64, 3), dtype=np.uint8)
        img_u8[10:20, 10:20] = 255
        canvas.set_processed_image(img_u8)
        pix_u8 = canvas.get_clipped_overlay_pixmap("highlights")
        self.assertIsNotNone(pix_u8)

        # 16-bit Mono
        img_u16 = np.zeros((50, 50), dtype=np.uint16)
        img_u16[5:15, 5:15] = 65535
        canvas.set_processed_image(img_u16)
        pix_u16 = canvas.get_clipped_overlay_pixmap("peaks")
        self.assertIsNotNone(pix_u16)

        # Float32 RGB [0.0..1.0] с пересветом
        img_f32 = np.full((80, 80, 3), 0.5, dtype=np.float32)
        img_f32[2:10, 2:10] = 1.0
        canvas.set_processed_image(img_f32)
        pix_f32 = canvas.get_clipped_overlay_pixmap("highlights")
        self.assertIsNotNone(pix_f32)

        # 2. Тест отрисовки drawForeground с активным сонаром (без падений)
        canvas.set_clipping_overlay_mode("highlights")
        self.assertGreater(len(canvas._clipping_centroids), 0)

        # Отрисовка на временном QImage
        qimg = QImage(200, 200, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(qimg)
        canvas.drawForeground(painter, QRectF(0, 0, 200, 200))
        painter.end()

        # Отключение
        canvas.set_clipping_overlay_mode(None)
        self.assertIsNone(canvas.clipping_overlay_mode)



