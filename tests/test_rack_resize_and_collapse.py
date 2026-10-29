"""Unit tests for Effect Inserts Rack interactive resizing, pinning, and collapsible tab handle."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QSizePolicy
from PyQt6.QtCore import Qt
from ui.main_window import AstroRackWindow
from ui.widgets import RackTabHandle

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class TestRackResizeAndCollapse(unittest.TestCase):
    def setUp(self):
        self.win = AstroRackWindow()
        self.win.show()
        QApplication.processEvents()

    def tearDown(self):
        self.win.close()
        QApplication.processEvents()

    def test_splitter_config_and_unlocked_width(self):
        """Пункт 1: Проверка настройки QSplitter и отсутствия жесткой блокировки ширины."""
        # Рэк не заблокирован по ширине и имеет диапазон [300, 550]
        self.assertEqual(self.win.right_panel.minimumWidth(), 300)
        self.assertEqual(self.win.right_panel.maximumWidth(), 550)

        # Коэффициенты растяжения (stretch factors): центральный вьювер забирает свободное место, рэк инсертов 0
        self.assertEqual(self.win.canvas_container.sizePolicy().horizontalStretch(), 1)
        self.assertEqual(self.win.right_panel.sizePolicy().horizontalStretch(), 0)

        # Интерактивный сплиттер не схлопывается
        self.assertFalse(self.win.splitter.isCollapsible(1))

        # Политика размеров Preferred, Expanding
        self.assertEqual(
            self.win.right_panel.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Preferred
        )
        self.assertEqual(
            self.win.right_panel.sizePolicy().verticalPolicy(),
            QSizePolicy.Policy.Expanding
        )

        # Начальная ширина 350
        self.assertEqual(self.win.right_panel.width(), 350)

    def test_interactive_resize_and_right_edge_pinning(self):
        """Пункт 2: Проверка интерактивного ресайза и жесткой фиксации правого края рэка."""
        total_w = self.win.splitter.width()
        self.assertGreater(total_w, 500)

        # Правая граница рэка строго равна ширине сплиттера
        self.assertEqual(self.win.right_panel.x() + self.win.right_panel.width(), total_w)

        # Увеличение ширины до 450
        self.win._set_panel_width(450)
        QApplication.processEvents()
        self.assertEqual(self.win.right_panel.width(), 450)
        self.assertEqual(self.win.right_panel.x() + self.win.right_panel.width(), total_w)

        # Уменьшение ширины до 320: координата X увеличивается, правый край остается на месте
        self.win._set_panel_width(320)
        QApplication.processEvents()
        self.assertEqual(self.win.right_panel.width(), 320)
        self.assertEqual(self.win.right_panel.x() + self.win.right_panel.width(), total_w)

        # Ограничение диапазона (clamping) [300, 550]
        self.win._set_panel_width(200)
        QApplication.processEvents()
        self.assertEqual(self.win.right_panel.width(), 300)

        self.win._set_panel_width(700)
        QApplication.processEvents()
        self.assertEqual(self.win.right_panel.width(), 550)

        # Тест перетаскивания через drag handle
        self.win.tab_handle._initial_panel_width = 350
        self.win._on_rack_handle_dragged(30)  # курсор сдвинулся вправо на 30px -> панель сужается до 320
        QApplication.processEvents()
        self.assertEqual(self.win.right_panel.width(), 320)
        self.assertEqual(self.win.right_panel.x() + self.win.right_panel.width(), total_w)

    def test_collapse_and_expand_unification(self):
        """Пункт 3: Проверка унификации сворачивания, шевронов и неоновой бирюзовой подсветки."""
        # Начальное состояние: развернут со старта приложения
        self.assertFalse(self.win._panel_collapsed)
        self.assertTrue(self.win.rack_inner_widget.isVisible())
        self.assertEqual(self.win.collapse_btn.text(), "⯈")
        self.assertEqual(self.win.tab_handle.collapse_btn.text(), "⯈")
        self.assertEqual(self.win.tab_handle.btn_toggle.text(), "⯈")
        self.assertEqual(self.win.collapse_btn.toolTip(), "Свернуть панель инсертов")
        self.assertIn("border-left: 1px solid #1a202c", self.win.tab_handle.styleSheet())
        # Проверка плоского минималистичного QSS идентичного нижней линейке (10px, #64748b, border none)
        btn_qss = self.win.collapse_btn.styleSheet()
        self.assertIn("background: transparent", btn_qss)
        self.assertIn("color: #64748b", btn_qss)
        self.assertIn("border: none", btn_qss)
        self.assertIn("font-size: 10px", btn_qss)

        # Сворачивание панели
        self.win._toggle_right_panel()
        QApplication.processEvents()

        self.assertTrue(self.win._panel_collapsed)
        self.assertFalse(self.win.rack_inner_widget.isVisible())
        self.assertEqual(self.win.right_panel.width(), 16)
        self.assertEqual(self.win.collapse_btn.text(), "⯇")
        self.assertEqual(self.win.collapse_btn.toolTip(), "Развернуть панель инсертов")
        self.assertIn("border-left: 2px solid #00e5ff", self.win.tab_handle.styleSheet())
        self.assertEqual(self.win.last_expanded_width, 350)

        # Разворачивание панели
        self.win._toggle_right_panel()
        QApplication.processEvents()

        self.assertFalse(self.win._panel_collapsed)
        self.assertTrue(self.win.rack_inner_widget.isVisible())
        self.assertEqual(self.win.right_panel.width(), 350)
        self.assertEqual(self.win.collapse_btn.text(), "⯈")
        self.assertEqual(self.win.collapse_btn.toolTip(), "Свернуть панель инсертов")
        self.assertIn("border-left: 1px solid #1a202c", self.win.tab_handle.styleSheet())

    def test_collapse_restores_custom_width(self):
        """Проверка восстановления пользовательской ширины после разворачивания."""
        # Пользователь настроил ширину 480
        self.win._set_panel_width(480)
        self.win.last_expanded_width = 480
        QApplication.processEvents()
        self.assertEqual(self.win.right_panel.width(), 480)

        # Сворачиваем
        self.win._toggle_right_panel()
        QApplication.processEvents()
        self.assertEqual(self.win.right_panel.width(), 16)
        self.assertEqual(self.win.last_expanded_width, 480)

        # Разворачиваем: ширина восстанавливается ровно до 480
        self.win._toggle_right_panel()
        QApplication.processEvents()
        self.assertEqual(self.win.right_panel.width(), 480)

    def test_insert_widgets_proper_height_and_scroll_area(self):
        """Проверка восстановления правильной высоты модулей и архитектуры ScrollArea."""
        # Проверка ScrollArea
        self.assertIsNotNone(self.win.scroll_area)
        self.assertTrue(self.win.scroll_area.widgetResizable())
        self.assertEqual(
            self.win.scroll_area.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.assertEqual(
            self.win.scroll_area.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        # Проверка выравнивания по верхнему краю (AlignTop)
        scroll_widget = self.win.scroll_area.widget()
        self.assertIsNotNone(scroll_widget)
        self.assertTrue(bool(scroll_widget.layout().alignment() & Qt.AlignmentFlag.AlignTop))

        # Проверка, что ни один инсерт не сплющен по вертикали (каждый выше 100px)
        inserts = [
            self.win.insert1_wavelet,
            self.win.insert2_rl,
            self.win.insert3_tone,
            self.win.insert4_mono,
            self.win.insert5_lc,
            self.win.insert6_color,
        ]
        for ins in inserts:
            self.assertGreater(
                ins.height(),
                100,
                f"{ins.__class__.__name__} is squashed vertically: height {ins.height()} <= 100"
            )
            self.assertEqual(
                ins.sizePolicy().verticalPolicy(),
                QSizePolicy.Policy.Minimum
            )

    def test_inserts_scroll_area_and_histogram_compactness(self):
        """Проверка устранения вылетания элементов за границы и настроек ScrollArea/Histogram."""
        # Наличие алиасов
        self.assertIs(self.win.inserts_scroll_area, self.win.scroll_area)
        self.assertIs(self.win.histogram_widget, self.win.histogram)

        # ScrollArea настройки
        self.assertTrue(self.win.inserts_scroll_area.widgetResizable())
        self.assertEqual(
            self.win.inserts_scroll_area.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # Histogram настройки
        self.assertLessEqual(self.win.histogram_widget.minimumWidth(), 240)
        self.assertEqual(
            self.win.histogram_widget.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Expanding
        )
        self.assertEqual(
            self.win.histogram_widget.sizePolicy().verticalPolicy(),
            QSizePolicy.Policy.Fixed
        )

        # При сжатии до 300px правая панель не выталкивается наружу
        self.win._set_panel_width(300)
        QApplication.processEvents()
        self.assertEqual(self.win.right_panel.width(), 300)
        total_w = self.win.splitter.width()
        self.assertEqual(self.win.right_panel.x() + self.win.right_panel.width(), total_w)



if __name__ == "__main__":
    unittest.main()
