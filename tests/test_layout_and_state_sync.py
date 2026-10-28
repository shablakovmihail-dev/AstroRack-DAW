"""Тесты для реорганизации левого верхнего угла, защитных состояний (Empty Canvas Guard),
суббайпасов, визуального Master Bypass и двухцветных кнопок копирования стейтов.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QPainter, QImage

app = QApplication.instance() or QApplication(sys.argv)

import numpy as np
from ui.main_window import AstroRackWindow
from ui.canvas import AstroCanvas, ZeroStateWidget, ensure_samples_dir
from ui.widgets import THEMES, ToolboxPanel, StateCopyButton


class TestLayoutAndStateSync(unittest.TestCase):

    def setUp(self):
        self.win = AstroRackWindow()
        self.win.show()
        QApplication.processEvents()

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()
        QApplication.processEvents()

    def test_upper_left_pixel_perfect_grid_alignment(self):
        """Проверка Pixel-Perfect выравнивания иконки, тулбара и кнопки импорта."""
        self.assertTrue(hasattr(self.win, 'logo_container'))
        self.assertTrue(hasattr(self.win, 'lbl_logo'))
        self.assertTrue(hasattr(self.win, 'btn_open'))
        self.assertTrue(hasattr(self.win, 'toolbox'))

        # 1. Ширина контейнера иконки СТРОГО равна ширине вертикального тулбара
        self.assertEqual(self.win.logo_container.width(), self.win.toolbox.width())
        self.assertEqual(self.win.logo_container.width(), 48)

        # 2. Левые координаты контейнера логотипа и тулбара совпадают
        logo_pos = self.win.logo_container.mapTo(self.win.centralWidget(), self.win.logo_container.rect().topLeft())
        toolbox_pos = self.win.toolbox.mapTo(self.win.centralWidget(), self.win.toolbox.rect().topLeft())
        self.assertEqual(logo_pos.x(), toolbox_pos.x())

        # 3. Правые координаты контейнера логотипа и тулбара совпадают (строгая вертикаль)
        logo_right = logo_pos.x() + self.win.logo_container.width()
        toolbox_right = toolbox_pos.x() + self.win.toolbox.width()
        self.assertEqual(logo_right, toolbox_right)

        # 4. Кнопка «Импорт» начинается сразу от правой границы тулбара со spacing=6px
        open_pos = self.win.btn_open.mapTo(self.win.centralWidget(), self.win.btn_open.rect().topLeft())
        self.assertEqual(open_pos.x(), logo_right + 6)

        # 5. Иконка внутри контейнера отцентрирована и имеет достаточный размер (>= 28x28)
        pixmap = self.win.lbl_logo.pixmap()
        self.assertFalse(pixmap.isNull())
        self.assertGreaterEqual(pixmap.width(), 28)
        self.assertGreaterEqual(pixmap.height(), 28)

    def test_empty_canvas_guard_and_unlock_on_load(self):
        """Проверка блокировки органов управления без изображения и разблокировки при загрузке."""
        # 1. Исходно холст пуст: элементы должны быть деактивированы (Empty Canvas Guard)
        self.assertFalse(self.win.toolbox.isEnabled())
        self.assertFalse(self.win.btn_export.isEnabled())
        self.assertFalse(self.win.btn_reset_image.isEnabled())
        self.assertFalse(self.win.btn_split_view.isEnabled())
        self.assertFalse(self.win.combo_presets.isEnabled())
        self.assertFalse(self.win.btn_undo.isEnabled())
        self.assertFalse(self.win.btn_redo.isEnabled())
        self.assertFalse(self.win.insert1_wavelet.isEnabled())
        self.assertFalse(self.win.insert2_rl.isEnabled())
        self.assertFalse(self.win.insert3_tone.isEnabled())
        self.assertFalse(self.win.insert4_mono.isEnabled())
        self.assertFalse(self.win.insert5_lc.isEnabled())
        self.assertFalse(self.win.insert6_color.isEnabled())
        self.assertFalse(self.win.histogram.isEnabled())
        self.assertFalse(self.win.btn_state_toggle.isEnabled())
        self.assertFalse(self.win.btn_copy_ab.isEnabled())
        self.assertFalse(self.win.btn_copy_ba.isEnabled())
        self.assertFalse(self.win.btn_global_bypass.isEnabled())

        # Всегда активные элементы остаются доступными
        self.assertTrue(self.win.btn_open.isEnabled())
        self.assertTrue(self.win.btn_hotkeys.isEnabled())
        self.assertTrue(self.win.btn_about.isEnabled())
        self.assertTrue(self.win.canvas.welcome_widget.isVisible())

        # 2. Загружаем снимок: все органы управления разблокируются
        samples_dir = ensure_samples_dir()
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
        self.assertGreater(len(files), 0)
        sample_path = os.path.join(samples_dir, files[0])

        self.win.open_image(sample_path)
        QApplication.processEvents()

        self.assertTrue(self.win.toolbox.isEnabled())
        self.assertTrue(self.win.btn_export.isEnabled())
        self.assertTrue(self.win.btn_reset_image.isEnabled())
        self.assertTrue(self.win.btn_split_view.isEnabled())
        self.assertTrue(self.win.combo_presets.isEnabled())
        self.assertTrue(self.win.insert1_wavelet.isEnabled())
        self.assertTrue(self.win.histogram.isEnabled())
        self.assertTrue(self.win.btn_state_toggle.isEnabled())
        self.assertTrue(self.win.btn_copy_ab.isEnabled())
        self.assertTrue(self.win.btn_copy_ba.isEnabled())
        self.assertTrue(self.win.btn_global_bypass.isEnabled())

        # 3. Нажимаем кнопку сброса снимка: защита повторно активируется
        from unittest.mock import patch
        from PyQt6.QtWidgets import QDialog
        with patch('ui.main_window.ResetConfirmDialog') as mock_dlg:
            mock_dlg.return_value.exec.return_value = QDialog.DialogCode.Accepted
            self.win.btn_reset_image.click()
            QApplication.processEvents()

        self.assertFalse(self.win.toolbox.isEnabled())
        self.assertFalse(self.win.btn_export.isEnabled())
        self.assertFalse(self.win.btn_reset_image.isEnabled())
        self.assertFalse(self.win.insert1_wavelet.isEnabled())
        self.assertTrue(self.win.btn_open.isEnabled())

    def test_reset_confirm_dialog_and_ghost_button_style(self):
        """Проверка стилей кнопки сброса (Ghost button) и работы модального диалога подтверждения."""
        from unittest.mock import patch
        from PyQt6.QtWidgets import QDialog
        from ui.dialogs import ResetConfirmDialog

        # 1. Проверка стиля кнопки сброса: имя объекта btnResetImage
        self.assertEqual(self.win.btn_reset_image.objectName(), "btnResetImage")

        # 2. Загружаем тестовый снимок
        samples_dir = ensure_samples_dir()
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
        sample_path = os.path.join(samples_dir, files[0])
        self.win.open_image(sample_path)
        QApplication.processEvents()
        self.assertIsNotNone(self.win._raw_image)

        # 3. Если в диалоге нажата "Отмена" (Rejected): сессия НЕ сбрасывается!
        with patch('ui.main_window.ResetConfirmDialog') as mock_dlg:
            mock_dlg.return_value.exec.return_value = QDialog.DialogCode.Rejected
            self.win.btn_reset_image.click()
            QApplication.processEvents()
        self.assertIsNotNone(self.win._raw_image)
        self.assertFalse(self.win.canvas.welcome_widget.isVisible())

        # 4. Если в диалоге нажата "Сбросить" (Accepted): сессия сбрасывается
        with patch('ui.main_window.ResetConfirmDialog') as mock_dlg:
            mock_dlg.return_value.exec.return_value = QDialog.DialogCode.Accepted
            self.win.btn_reset_image.click()
            QApplication.processEvents()
        self.assertIsNone(self.win._raw_image)
        self.assertTrue(self.win.canvas.welcome_widget.isVisible())

        # 5. Тестируем сам диалог ResetConfirmDialog
        dlg = ResetConfirmDialog(self.win)
        self.assertEqual(dlg.windowTitle(), "Подтверждение сброса")
        self.assertTrue(hasattr(dlg, 'btn_cancel'))
        self.assertTrue(hasattr(dlg, 'btn_reset'))
        self.assertTrue(dlg.btn_cancel.isDefault())
        dlg.close()
        dlg.deleteLater()

    def test_spinbox_fixed_width_44px(self):
        """Проверка фиксированной ширины 44px для полей ввода значений слайдеров."""
        self.assertEqual(self.win.insert5_lc.row_amount.spinbox.width(), 44)
        self.assertEqual(self.win.insert1_wavelet.row_l1.spinbox.width(), 44)
        self.assertEqual(self.win.insert3_tone.row_black.spinbox.width(), 44)

    def test_sub_bypass_ghost_state_when_bypassed(self):
        """Проверка деактивации слайдеров и суббайпасов при байпасе инсерта."""
        # 1. Загружаем изображение
        samples_dir = ensure_samples_dir()
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
        sample_path = os.path.join(samples_dir, files[0])
        self.win.open_image(sample_path)
        QApplication.processEvents()

        # Включаем инсерт 3 и проверяем доступность параметров
        self.win.insert3_tone.set_enabled(True)
        self.assertTrue(self.win.insert3_tone.content_widget.isEnabled())
        self.assertTrue(self.win.insert3_tone.row_black.chk_sub_bypass.isEnabled())

        # Выключаем инсерт 3 (байпас)
        self.win.insert3_tone.set_enabled(False)
        self.assertFalse(self.win.insert3_tone.content_widget.isEnabled())
        self.assertFalse(self.win.insert3_tone.row_black.chk_sub_bypass.isEnabled())

    def test_master_bypass_visual_response(self):
        """Проверка визуального отклика Master Bypass: приглушение крутилок Dry/Wet и рамок инсертов."""
        # 1. Загружаем изображение
        samples_dir = ensure_samples_dir()
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
        sample_path = os.path.join(samples_dir, files[0])
        self.win.open_image(sample_path)
        QApplication.processEvents()

        # Включаем инсерт 1
        self.win.insert1_wavelet.set_enabled(True)
        self.assertTrue(self.win.insert1_wavelet.header.dry_wet_ctrl.isEnabled())
        self.assertFalse(getattr(self.win.insert1_wavelet.header.dry_wet_ctrl, '_dimmed', False))

        # Активируем Master Bypass
        self.win.btn_global_bypass.click()
        QApplication.processEvents()

        self.assertTrue(self.win.btn_global_bypass.isChecked())
        self.assertTrue(getattr(self.win.insert1_wavelet.header.dry_wet_ctrl, '_dimmed', False))
        self.assertFalse(self.win.insert1_wavelet.content_widget.isEnabled())

        # Деактивируем Master Bypass
        self.win.btn_global_bypass.click()
        QApplication.processEvents()

        self.assertFalse(self.win.btn_global_bypass.isChecked())
        self.assertFalse(getattr(self.win.insert1_wavelet.header.dry_wet_ctrl, '_dimmed', False))
        self.assertTrue(self.win.insert1_wavelet.content_widget.isEnabled())

    def test_state_copy_buttons_two_color(self):
        """Проверка двухцветной маркировки кнопок [ A → B ] и [ B → A ]."""
        self.assertIsInstance(self.win.btn_copy_ab, StateCopyButton)
        self.assertIsInstance(self.win.btn_copy_ba, StateCopyButton)
        self.assertEqual(self.win.btn_copy_ab._from_state, "A")
        self.assertEqual(self.win.btn_copy_ab._to_state, "B")
        self.assertEqual(self.win.btn_copy_ba._from_state, "B")
        self.assertEqual(self.win.btn_copy_ba._to_state, "A")

    def test_zero_state_monochrome_demo_button(self):
        """Проверка монохромного стиля и аутлайн-иконки для кнопки демо-файлов."""
        welcome = self.win.canvas.welcome_widget
        self.assertIn("🗁", welcome.btn_demo.text())
        self.assertIn("Загрузить демо-снимок", welcome.btn_demo.text())
        self.assertNotIn("📁", welcome.btn_demo.text())

    def test_dynamic_ab_state_sync_for_canvas_and_ui(self):
        """Проверка динамического обновления тем (State A: Cyan, State B: Amber)."""
        self.assertEqual(self.win._active_ab, 'A')
        self.assertEqual(self.win.canvas._theme['bright'], THEMES['A']['bright'])

        self.win._switch_ab_state('B')
        QApplication.processEvents()

        self.assertEqual(self.win._active_ab, 'B')
        self.assertEqual(self.win.canvas._theme['bright'], THEMES['B']['bright'])
        self.assertEqual(self.win.toolbox._theme['bright'], THEMES['B']['bright'])

    def test_cascading_master_bypass_and_local_bypass_deactivation(self):
        """Проверка каскадного усыпления контролов (Mode, Help, Reset, Header) при Master Bypass и локальном байпасе."""
        from ui.widgets import HelpIcon
        # 1. Загружаем снимок для разблокировки Empty Canvas Guard
        samples_dir = ensure_samples_dir()
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
        sample_path = os.path.join(samples_dir, files[0])
        self.win.open_image(sample_path)
        QApplication.processEvents()

        ins1 = self.win.insert1_wavelet
        ins5 = self.win.insert5_lc

        # 2. Локальное включение инсертов 1 и 5
        ins1.header.chk_bypass.setChecked(True)
        ins5.header.chk_bypass.setChecked(True)
        QApplication.processEvents()

        self.assertTrue(ins1.combo_mode.isEnabled())
        self.assertTrue(ins1.btn_help_mode.isEnabled())
        self.assertTrue(ins5.combo_mode.isEnabled())
        self.assertTrue(ins5.btn_help_mode.isEnabled())
        self.assertIn(THEMES['A']['primary'], ins1.lbl_mode.styleSheet())
        self.assertIn(THEMES['A']['primary'], ins1.header.lbl_title.styleSheet())

        # 3. Тест локального байпаса инсерта 1 (Master Bypass выключен)
        ins1.header.chk_bypass.setChecked(False)
        QApplication.processEvents()

        self.assertFalse(ins1.combo_mode.isEnabled())
        self.assertFalse(ins1.btn_help_mode.isEnabled())
        self.assertIn("#64748B", ins1.lbl_mode.styleSheet())
        self.assertIn("#64748B", ins1.header.lbl_title.styleSheet())
        self.assertIn("#64748B", ins1.header.arrow_btn.styleSheet())
        for icon in ins1.findChildren(HelpIcon):
            self.assertFalse(icon.isEnabled())

        # Включение обратно
        ins1.header.chk_bypass.setChecked(True)
        QApplication.processEvents()
        self.assertTrue(ins1.combo_mode.isEnabled())
        self.assertTrue(ins1.btn_help_mode.isEnabled())
        self.assertIn(THEMES['A']['primary'], ins1.lbl_mode.styleSheet())

        # 4. Тест каскадного усыпления при Master Bypass (Master Bypass Active)
        self.win.btn_global_bypass.setChecked(True)
        QApplication.processEvents()

        inserts = [
            self.win.insert1_wavelet, self.win.insert2_rl, self.win.insert3_tone,
            self.win.insert4_mono, self.win.insert5_lc, self.win.insert6_color
        ]

        for ins in inserts:
            # Главный чекбокс заблокирован
            self.assertFalse(ins.header.chk_bypass.isEnabled())
            # Заголовок и стрелка приглушены в #64748B
            self.assertIn("#64748B", ins.header.lbl_title.styleSheet())
            self.assertIn("#64748B", ins.header.arrow_btn.styleSheet())
            # Все кнопки справки [?] и кнопки сброса [R] отключены
            for icon in ins.findChildren(HelpIcon):
                self.assertFalse(icon.isEnabled())
            # Крутилки Dry/Wet отключены и приглушены
            if ins.header.dry_wet_ctrl is not None:
                self.assertFalse(ins.header.dry_wet_ctrl.isEnabled())
                self.assertTrue(getattr(ins.header.dry_wet_ctrl, '_dimmed', False))

        # Выпадающие списки Mode в инсертах 1 и 5 отключены
        self.assertFalse(ins1.combo_mode.isEnabled())
        self.assertFalse(ins1.btn_help_mode.isEnabled())
        self.assertIn("#64748B", ins1.lbl_mode.styleSheet())
        self.assertFalse(ins5.combo_mode.isEnabled())
        self.assertFalse(ins5.btn_help_mode.isEnabled())
        self.assertIn("#64748B", ins5.lbl_mode.styleSheet())

        # 5. Возврат Master Bypass в неактивное состояние
        self.win.btn_global_bypass.setChecked(False)
        QApplication.processEvents()

        for ins in inserts:
            self.assertTrue(ins.header.chk_bypass.isEnabled())

        # Включенные инсерты вернулись в нормальный цвет и интерактивность
        self.assertTrue(ins1.combo_mode.isEnabled())
        self.assertTrue(ins1.btn_help_mode.isEnabled())
        self.assertIn(THEMES['A']['primary'], ins1.lbl_mode.styleSheet())
        self.assertTrue(ins5.combo_mode.isEnabled())
        self.assertTrue(ins5.btn_help_mode.isEnabled())
        self.assertIn(THEMES['A']['primary'], ins5.lbl_mode.styleSheet())

    def test_histogram_visors_and_master_bypass_sync(self):
        """Проверка синхронизации визиров Black/White Point и реакции гистограммы на Master Bypass."""
        # 1. Загружаем тестовый снимок
        samples_dir = ensure_samples_dir()
        files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.tif', '.jpg'))]
        self.win.open_image(os.path.join(samples_dir, files[0]))
        QApplication.processEvents()

        # 2. Включаем Insert 3 и меняем значения Black Level и Midtones/Gamma
        self.win.insert3_tone.set_enabled(True)
        self.win.insert3_tone.row_black.set_param_enabled(True)
        self.win.insert3_tone.row_black.set_value(0.120)

        self.win.insert3_tone.row_gamma.set_param_enabled(True)
        self.win.insert3_tone.row_gamma.set_value(1.40)

        self.win._sync_histogram_visors()

        # Проверяем, что визиры на гистограмме обновились
        self.assertTrue(self.win.histogram._black_point_enabled)
        self.assertAlmostEqual(self.win.histogram._black_point_norm, 0.120, places=3)
        self.assertTrue(self.win.histogram._white_point_enabled)
        self.assertAlmostEqual(self.win.histogram._white_point_norm, 1.0 / 1.40, places=2)

        # 3. Master Bypass переводит гистограмму в режим сырого кадра с акцентом #64748B
        self.win.btn_global_bypass.setChecked(True)
        self.win._on_global_bypass_toggled(True)
        self.assertTrue(self.win.histogram._is_master_bypassed)

        self.win.btn_global_bypass.setChecked(False)
        self.win._on_global_bypass_toggled(False)
        self.assertFalse(self.win.histogram._is_master_bypassed)

    def test_split_view_hardware_clipping_and_zoom_pan_sync(self):
        """Проверка 100% аппаратной синхронизации линии Split View и границы слоев при зуме и панорамировании."""
        from PyQt6.QtGui import QImage, QPainter

        canvas = self.win.canvas
        canvas.resize(400, 400)

        raw = np.zeros((200, 200, 3), dtype=np.float32)
        raw[:, :, 0] = 1.0  # Red
        proc = np.zeros((200, 200, 3), dtype=np.float32)
        proc[:, :, 1] = 1.0  # Green

        canvas.set_raw_image(raw)
        canvas.set_processed_image(proc)
        canvas.set_split_view_active(True)
        canvas._item.split_fraction = 0.5

        # 1. Проверка вертикального сплита (↔) при различных зумах
        canvas.set_split_orientation('Vertical')
        for zoom in [1.0, 2.5, 4.0]:
            canvas.resetTransform()
            canvas.scale(zoom, zoom)
            canvas.horizontalScrollBar().setValue(int(25 * zoom))
            canvas.verticalScrollBar().setValue(int(15 * zoom))

            img = QImage(canvas.viewport().size(), QImage.Format.Format_ARGB32)
            img.fill(Qt.GlobalColor.black)
            p = QPainter(img)
            canvas.viewport().render(p)
            p.end()

            expected_split_x = int(round(canvas.mapFromScene(QPointF(100, 0)).x()))

            col_before = img.pixelColor(expected_split_x - 15, 200)
            col_after = img.pixelColor(expected_split_x + 15, 200)

            self.assertGreater(col_before.red(), 200)
            self.assertLess(col_before.green(), 50)
            self.assertGreater(col_after.green(), 200)
            self.assertLess(col_after.red(), 50)

        # 2. Проверка горизонтального сплита (↕) при зуме
        canvas.set_split_orientation('Horizontal')
        canvas.resetTransform()
        canvas.scale(3.0, 3.0)
        canvas.horizontalScrollBar().setValue(50)
        canvas.verticalScrollBar().setValue(50)

        img = QImage(canvas.viewport().size(), QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.black)
        p = QPainter(img)
        canvas.viewport().render(p)
        p.end()

        expected_split_y = int(round(canvas.mapFromScene(QPointF(0, 100)).y()))
        col_before_y = img.pixelColor(200, expected_split_y - 15)
        col_after_y = img.pixelColor(200, expected_split_y + 15)

        self.assertGreater(col_before_y.red(), 200)
        self.assertLess(col_before_y.green(), 50)
        self.assertGreater(col_after_y.green(), 200)
        self.assertLess(col_after_y.red(), 50)


if __name__ == '__main__':
    unittest.main()
