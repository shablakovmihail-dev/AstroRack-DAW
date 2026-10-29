import unittest
import datetime
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPointF, QDate
from PyQt6.QtGui import QPixmap

from ui.widgets import TelescopeOpticsDialog, ToolboxPanel
from ui.canvas import AstroCanvas
from ui.measurement_hud import MeasurementHUD, MeasurementHelpDialog
from core.lunar_distance import calculate_lunar_distance
from core.geometry_calc import fit_circle_3points, calculate_spherical_correction, calculate_crater_ellipse

# Инициализируем QApplication для тестов виджетов Qt
app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestLunarRulerUI(unittest.TestCase):
    def test_optics_dialog_default_initialization(self):
        dlg = TelescopeOpticsDialog(
            focal_len=1000.0,
            pixel_size=3.76,
            moon_distance_km=384400.0,
            lunar_mode="date",
            is_auto_detected=False
        )
        self.assertEqual(dlg.spin_f.value(), 1000.0)
        self.assertEqual(dlg.spin_p.value(), 3.76)
        self.assertTrue(dlg.radio_by_date.isChecked())
        self.assertFalse(dlg.radio_manual.isChecked())
        self.assertTrue(dlg.date_edit.isEnabled())
        self.assertFalse(dlg.distance_spinbox.isEnabled())
        self.assertTrue(dlg.lbl_badge.isHidden())

        vals = dlg.get_values()
        self.assertEqual(len(vals), 3)
        self.assertEqual(vals[0], 1000.0)
        self.assertEqual(vals[1], 3.76)

    def test_optics_dialog_auto_detected_badge(self):
        capture_dt = datetime.datetime(2023, 8, 30, 21, 0, 0)
        dlg = TelescopeOpticsDialog(
            focal_len=800.0,
            pixel_size=2.90,
            capture_date=capture_dt,
            is_auto_detected=True,
            lunar_mode="date"
        )
        self.assertFalse(dlg.lbl_badge.isHidden())
        self.assertEqual(dlg.date_edit.date(), QDate(2023, 8, 30))
        # Проверяем, что дистанция рассчиталась автоматически по дате
        expected_dist = calculate_lunar_distance(datetime.date(2023, 8, 30), round_digits=1)
        self.assertAlmostEqual(dlg.distance_spinbox.value(), expected_dist, delta=0.5)

    def test_optics_dialog_manual_mode_switch(self):
        dlg = TelescopeOpticsDialog(
            focal_len=1200.0,
            pixel_size=3.76,
            moon_distance_km=390000.0,
            lunar_mode="date",
            is_auto_detected=True
        )
        # Переключаем в ручной режим
        dlg.radio_manual.setChecked(True)
        self.assertFalse(dlg.date_edit.isEnabled())
        self.assertTrue(dlg.distance_spinbox.isEnabled())
        self.assertTrue(dlg.lbl_badge.isHidden())
        self.assertEqual(dlg.get_lunar_mode(), "manual")

        # Вводим вручную
        dlg.distance_spinbox.setValue(395000.0)
        self.assertEqual(dlg.get_moon_distance(), 395000.0)

        # Переключаем обратно в режим по дате
        dlg.radio_by_date.setChecked(True)
        self.assertTrue(dlg.date_edit.isEnabled())
        self.assertFalse(dlg.distance_spinbox.isEnabled())
        self.assertFalse(dlg.lbl_badge.isHidden())
        self.assertEqual(dlg.get_lunar_mode(), "date")

    def test_optics_dialog_preview_callback(self):
        preview_events = []

        def on_preview(f, p, dist):
            preview_events.append((f, p, dist))

        dlg = TelescopeOpticsDialog(
            focal_len=1000.0,
            pixel_size=3.76,
            on_preview_update=on_preview
        )
        dlg.spin_f.setValue(1500.0)
        self.assertTrue(len(preview_events) > 0)
        last_f, last_p, _ = preview_events[-1]
        self.assertEqual(last_f, 1500.0)
        self.assertEqual(last_p, 3.76)

    def test_canvas_ruler_formula_with_lunar_distance(self):
        canvas = AstroCanvas()
        canvas.set_telescope_params(1000.0, 3.76)  # F = 1000 mm, Pixel = 3.76 um
        canvas.set_ruler_unit("km")

        # Проверяем для двух разных расстояний до Луны (перигей ~360 000 vs апогей ~405 000)
        dist_px = 100.0

        # Перигей
        canvas.set_lunar_distance(360000.0)
        dist_str_perigee, _ = canvas._format_ruler_dist(dist_px)
        # km = (100 * (3.76 / (1000 * 1000))) * 360000 = 100 * 0.00000376 * 360000 = 135.36 km
        self.assertIn("135.4 km", dist_str_perigee)

        # Апогей
        canvas.set_lunar_distance(405000.0)
        dist_str_apogee, _ = canvas._format_ruler_dist(dist_px)
        # km = 100 * 0.00000376 * 405000 = 152.28 km
        self.assertIn("152.3 km", dist_str_apogee)

        # Разница должна составить около ~16.9 км (~12.5% от перигея к апогею)
        self.assertNotEqual(dist_str_perigee, dist_str_apogee)

    def test_canvas_ruler_redraw_on_lunar_distance_change(self):
        canvas = AstroCanvas()
        canvas.set_telescope_params(1000.0, 3.76)
        canvas.set_ruler_unit("km")

        # Устанавливаем линейку
        canvas._ruler_p1 = QPointF(0, 0)
        canvas._ruler_p2 = QPointF(100, 0)

        measured_summaries = []
        canvas.rulerMeasured.connect(lambda dist, angle, summary: measured_summaries.append(summary))

        # Изменение расстояния должно мгновенно пересчитывать summary и эмитить сигнал
        canvas.set_lunar_distance(370000.0)
        self.assertTrue(len(measured_summaries) > 0)
        self.assertIn("139.1 km", measured_summaries[-1])

        canvas.set_lunar_distance(400000.0)
        self.assertIn("150.4 km", measured_summaries[-1])

    def test_measurement_hud_initialization_and_mode_switching(self):
        canvas = AstroCanvas()
        self.assertIsNotNone(canvas.measurement_hud)

        modes_received = []
        canvas.rulerSubmodeChanged.connect(modes_received.append)

        # Switch to disk alignment mode
        canvas.set_measurement_submode("disk")
        self.assertEqual(canvas._meas_submode, "disk")
        self.assertEqual(canvas.measurement_hud._current_mode, "disk")
        self.assertIn("disk", modes_received)

        # Switch to ellipse mode
        canvas.set_measurement_submode("ellipse")
        self.assertEqual(canvas._meas_submode, "ellipse")
        self.assertEqual(canvas.measurement_hud._current_mode, "ellipse")
        self.assertIn("ellipse", modes_received)

        # Switch back to linear ruler
        canvas.set_measurement_submode("ruler")
        self.assertEqual(canvas._meas_submode, "ruler")
        self.assertEqual(canvas.measurement_hud._current_mode, "ruler")

    def test_canvas_disk_calibration_center_radius(self):
        canvas = AstroCanvas()
        canvas.set_measurement_submode("disk")
        canvas.set_disk_method("center_radius")

        # Simulate center + radius draft
        canvas._disk_draft_center = QPointF(500.0, 500.0)
        canvas._disk_draft_radius = 450.0
        canvas.apply_disk_alignment()

        self.assertIsNotNone(canvas._calibrated_disk)
        self.assertEqual(canvas._calibrated_disk, (500.0, 500.0, 450.0))
        self.assertTrue(canvas.measurement_hud.btn_reset_disk.isEnabled())
        self.assertIn("450 px", canvas.measurement_hud.lbl_disk_status.text())

        # Reset disk alignment
        canvas.reset_disk_alignment()
        self.assertIsNone(canvas._calibrated_disk)
        self.assertFalse(canvas.measurement_hud.btn_reset_disk.isEnabled())
        self.assertTrue(canvas.measurement_hud.disk_status_box.isHidden())

    def test_canvas_disk_calibration_three_points(self):
        canvas = AstroCanvas()
        canvas.set_measurement_submode("disk")
        canvas.set_disk_method("three_points")

        # 3 points on circle centered at (200, 300) with R=100
        canvas._disk_points = [
            QPointF(200.0, 400.0),  # top
            QPointF(300.0, 300.0),  # right
            QPointF(200.0, 200.0)   # bottom
        ]
        canvas.apply_disk_alignment()

        self.assertIsNotNone(canvas._calibrated_disk)
        cx, cy, r = canvas._calibrated_disk
        self.assertAlmostEqual(cx, 200.0, delta=0.5)
        self.assertAlmostEqual(cy, 300.0, delta=0.5)
        self.assertAlmostEqual(r, 100.0, delta=0.5)

    def test_spherical_correction_in_ruler_results(self):
        canvas = AstroCanvas()
        canvas.set_telescope_params(1000.0, 3.76)
        canvas.set_lunar_distance(384400.0)

        # Calibrated disk at (500, 500) with R=500
        canvas._calibrated_disk = (500.0, 500.0, 500.0)

        # Segment near center: (500, 500) -> (550, 500) (length = 50 px)
        canvas._ruler_p1 = QPointF(500.0, 500.0)
        canvas._ruler_p2 = QPointF(550.0, 500.0)
        canvas._emit_ruler_info()

        # Near center: cos(theta) ~ 1.0, True km ~ Proj km
        scale_km = (3.76 / (1000.0 * 1000.0)) * 384400.0
        proj_km_center = 50.0 * scale_km
        self.assertAlmostEqual(canvas.measurement_hud._last_proj_km, proj_km_center, delta=0.1)
        self.assertAlmostEqual(canvas.measurement_hud._last_true_km, proj_km_center, delta=1.5)

        # Segment near limb: (900, 500) -> (950, 500) (length = 50 px, radial distance ~425 px / 500 px => cos θ ~ 0.5)
        canvas._ruler_p1 = QPointF(900.0, 500.0)
        canvas._ruler_p2 = QPointF(950.0, 500.0)
        canvas._emit_ruler_info()

        # True km must be significantly greater than Proj km due to perspective foreshortening
        self.assertGreater(canvas.measurement_hud._last_true_km, canvas.measurement_hud._last_proj_km * 1.5)

    def test_crater_ellipse_workflow_and_hud(self):
        canvas = AstroCanvas()
        canvas.set_telescope_params(1000.0, 3.76)
        canvas.set_lunar_distance(384400.0)
        canvas.set_measurement_submode("ellipse")

        # Set major axis from (100, 200) to (200, 200) => length = 100, semi-major a = 50
        canvas._ellipse_p_start = QPointF(100.0, 200.0)
        canvas._ellipse_p_end = QPointF(200.0, 200.0)

        # Cursor at (150, 230) => perpendicular distance = 30, semi-minor b = 30
        canvas._update_ellipse_calc(QPointF(150.0, 230.0))

        self.assertIsNotNone(canvas._ellipse_data)
        d = canvas._ellipse_data
        self.assertEqual(d["semi_major_a"], 50.0)
        self.assertEqual(d["semi_minor_b"], 30.0)
        self.assertGreater(d["true_diameter_km"], 0.0)
        self.assertGreater(d["apparent_diameter_km"], 0.0)
        self.assertAlmostEqual(d["cos_theta"], 0.6, delta=0.01)

        # Check HUD labels
        self.assertIn("True Dia", canvas.measurement_hud.row_res_1.text())
        self.assertIn("Apparent", canvas.measurement_hud.row_res_2.text())
        self.assertIn("Tilt", canvas.measurement_hud.row_res_3.text())

    def test_toolbox_panel_measurement_submode_signal(self):
        toolbox = ToolboxPanel()
        submode_events = []
        toolbox.ruler_submode_changed.connect(submode_events.append)

        toolbox._set_ruler_submode("disk")
        self.assertEqual(toolbox._ruler_submode, "disk")
        self.assertIn("disk", submode_events)

        toolbox._set_ruler_submode("ellipse")
        self.assertEqual(toolbox._ruler_submode, "ellipse")
        self.assertIn("ellipse", submode_events)

    def test_hud_drop_shadow_and_borders(self):
        hud = MeasurementHUD()
        self.assertEqual(hud.height(), 60)
        self.assertTrue(hud.is_expanded)
        self.assertIn("background-color: #14171d", hud.styleSheet())
        self.assertIn("border-top: 1px solid #1a202c", hud.styleSheet())

        self.assertIn("border: 1px solid #2a3441", hud.disk_status_box.styleSheet())
        self.assertIn("border-radius: 6px", hud.disk_status_box.styleSheet())
        self.assertTrue(hud.results_box.isHidden())

        # Test collapse toggle
        close_events = []
        hud.close_requested.connect(lambda: close_events.append(True))
        hud.btn_close.click()
        self.assertTrue(len(close_events) > 0)
        self.assertTrue(hud.is_collapsed)
        self.assertFalse(hud.is_expanded)
        self.assertEqual(hud.height(), 14)
        self.assertTrue(hud.shelf_body.isHidden())
        self.assertIn("border-top: 2px solid #00e5ff", hud.tab_handle.styleSheet())

        # Click again to restore
        hud.btn_close.click()
        self.assertFalse(hud.is_collapsed)
        self.assertTrue(hud.is_expanded)
        self.assertEqual(hud.height(), 60)
        self.assertFalse(hud.shelf_body.isHidden())
        self.assertIn("border-top: 1px solid #1a202c", hud.tab_handle.styleSheet())

    def test_figma_bounding_box_drag_and_1to1_constraint(self):
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QPointF, Qt

        canvas = AstroCanvas()
        pix = QPixmap(1000, 1000)
        canvas._item.processed_pixmap = pix

        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")

        # Press at (100, 100)
        ev_press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(100.0, 100.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev_press)
        self.assertTrue(canvas.disk_is_dragging)
        self.assertIsNotNone(canvas.disk_drag_start)

        # Drag to (300, 100) along diameter -> diameter = 200 px, radius = 100 px
        ev_move = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(300.0, 100.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseMoveEvent(ev_move)

        # Center should be at midpoint (200, 100), radius = 100
        self.assertEqual(canvas.disk_radius, 100.0)
        self.assertIsNotNone(canvas.temp_disk_rect)
        self.assertEqual(canvas.temp_disk_rect.width(), 200.0)
        self.assertEqual(canvas.temp_disk_rect.height(), 200.0)
        self.assertEqual(canvas.disk_center, canvas.temp_disk_rect.center())

        # Release mouse -> sets pending disk
        ev_rel = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(300.0, 100.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseReleaseEvent(ev_rel)
        self.assertFalse(canvas.disk_is_dragging)
        self.assertIsNotNone(canvas.pending_disk)
        self.assertEqual(canvas.pending_disk[2], 100.0)

    def test_figma_handle_transform_and_cursors(self):
        from PyQt6.QtGui import QMouseEvent, QKeyEvent
        from PyQt6.QtCore import QPointF, Qt, QPoint

        canvas = AstroCanvas()
        pix = QPixmap(1000, 1000)
        canvas._item.processed_pixmap = pix

        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")

        # Set disk at center (500, 500) with R=100
        canvas.disk_center = QPointF(500.0, 500.0)
        canvas._disk_draft_center = canvas.disk_center
        canvas.disk_radius = 100.0
        canvas._disk_draft_radius = 100.0
        canvas._disk_state = "editing"

        # Viewport position of center is mapFromScene(500, 500)
        c_vp = canvas.mapFromScene(canvas.disk_center)
        self.assertEqual(canvas._hit_test_disk_handles(c_vp), "center")
        self.assertEqual(canvas._cursor_for_disk_handle("center"), Qt.CursorShape.SizeAllCursor)

        # Bottom-right corner is (600, 600)
        br_vp = canvas.mapFromScene(QPointF(600.0, 600.0))
        self.assertEqual(canvas._hit_test_disk_handles(br_vp), "br")
        self.assertEqual(canvas._cursor_for_disk_handle("br"), Qt.CursorShape.SizeFDiagCursor)

        # Top-right corner is (600, 400)
        tr_vp = canvas.mapFromScene(QPointF(600.0, 400.0))
        self.assertEqual(canvas._hit_test_disk_handles(tr_vp), "tr")
        self.assertEqual(canvas._cursor_for_disk_handle("tr"), Qt.CursorShape.SizeBDiagCursor)

    def test_figma_keyboard_fine_adjustment(self):
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import QPointF, Qt

        canvas = AstroCanvas()
        pix = QPixmap(1000, 1000)
        canvas._item.processed_pixmap = pix

        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")

        canvas.disk_center = QPointF(500.0, 500.0)
        canvas._disk_draft_center = canvas.disk_center
        canvas.disk_radius = 100.0
        canvas._disk_draft_radius = 100.0
        canvas._disk_state = "editing"

        # 1. Right arrow -> shifts center x by +1 px
        ev_right = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(ev_right)
        self.assertEqual(canvas.disk_center.x(), 501.0)
        self.assertEqual(canvas.disk_center.y(), 500.0)

        # 2. Shift + Down arrow -> shifts center y by +10 px
        ev_down_shift = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
        canvas.keyPressEvent(ev_down_shift)
        self.assertEqual(canvas.disk_center.x(), 501.0)
        self.assertEqual(canvas.disk_center.y(), 510.0)

        # 3. Plus key -> expands radius by +1 px
        ev_plus = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Plus, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(ev_plus)
        self.assertEqual(canvas.disk_radius, 101.0)

        # 4. Shift + Plus key -> expands radius by +5 px
        ev_plus_shift = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Plus, Qt.KeyboardModifier.ShiftModifier)
        canvas.keyPressEvent(ev_plus_shift)
        self.assertEqual(canvas.disk_radius, 106.0)

        # 5. Minus key -> decreases radius by -1 px
        ev_minus = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Minus, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(ev_minus)
        self.assertEqual(canvas.disk_radius, 105.0)

    def test_figma_apply_and_auto_switch_to_ruler(self):
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import QPointF, Qt

        canvas = AstroCanvas()
        pix = QPixmap(1000, 1000)
        canvas._item.processed_pixmap = pix

        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")

        canvas.disk_center = QPointF(500.0, 500.0)
        canvas._disk_draft_center = canvas.disk_center
        canvas.disk_radius = 120.0
        canvas._disk_draft_radius = 120.0
        canvas._disk_state = "editing"

        # Press Enter
        ev_enter = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(ev_enter)

        # Calibrated disk should be set
        self.assertEqual(canvas._calibrated_disk, (500.0, 500.0, 120.0))
        # Status should show calibrated
        self.assertTrue(canvas.measurement_hud.btn_reset_disk.isEnabled())
        self.assertIn("120 px", canvas.measurement_hud.lbl_disk_status.text())
        # Automatically switched back to ruler mode!
        self.assertEqual(canvas._meas_submode, "ruler")
        self.assertEqual(canvas.measurement_hud._current_mode, "ruler")

    def test_canvas_state_structures_initialized(self):
        canvas = AstroCanvas()
        self.assertIsInstance(canvas.disk_points, list)
        self.assertIsNone(canvas.disk_center)
        self.assertEqual(canvas.disk_radius, 0.0)
        self.assertFalse(canvas.is_drawing_disk)

        self.assertIsNone(canvas.ellipse_center)
        self.assertEqual(canvas.ellipse_a, 0.0)
        self.assertEqual(canvas.ellipse_b, 0.0)
        self.assertEqual(canvas.ellipse_angle, 0.0)
        self.assertEqual(canvas.ellipse_stage, 0)
        self.assertEqual(canvas.active_mode, "ruler")

    def test_canvas_click_safety_center_radius(self):
        canvas = AstroCanvas()
        # Create dummy pixmap so canvas has valid image bounds
        pix = QPixmap(500, 500)
        canvas._item.processed_pixmap = pix

        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")
        canvas.set_disk_method("center_radius")

        # Simulate click 1 in center_radius mode
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QPointF, Qt, QPoint
        event1 = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(250.0, 250.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier
        )
        canvas.mousePressEvent(event1)

        # Verify state: center set, is_drawing_disk = True, radius is 0, no exception
        self.assertIsNotNone(canvas.disk_center)
        self.assertTrue(canvas.is_drawing_disk)
        self.assertEqual(canvas.disk_radius, 0.0)

    def test_canvas_click_safety_three_points(self):
        canvas = AstroCanvas()
        pix = QPixmap(500, 500)
        canvas._item.processed_pixmap = pix

        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")
        canvas.set_disk_method("three_points")

        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QPointF, Qt

        # Point 1
        ev1 = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(100.0, 100.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev1)
        self.assertEqual(len(canvas.disk_points), 1)
        self.assertIsNone(canvas.disk_center)

        # Point 2
        ev2 = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(200.0, 100.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev2)
        self.assertEqual(len(canvas.disk_points), 2)
        self.assertIsNone(canvas.disk_center)

        # Point 3 collinear with p1 and p2 -> fit_circle fails safely, point 3 popped, app doesn't crash!
        ev3 = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(300.0, 100.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev3)
        self.assertEqual(len(canvas.disk_points), 2)
        self.assertIsNone(canvas.disk_center)

    def test_canvas_click_safety_ellipse(self):
        canvas = AstroCanvas()
        pix = QPixmap(500, 500)
        canvas._item.processed_pixmap = pix

        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("ellipse")

        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QPointF, Qt

        # Click to start ellipse (stage 1)
        ev1 = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(100.0, 100.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev1)
        self.assertEqual(canvas.ellipse_stage, 1)
        self.assertIsNotNone(canvas.ellipse_start)

        # Zero length drag release -> doesn't crash, resets safely
        ev_rel = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(100.0, 100.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseReleaseEvent(ev_rel)
        self.assertEqual(canvas.ellipse_stage, 0)

    def test_layout_grid_isolation_and_shelf_compaction(self):
        from ui.main_window import AstroRackWindow
        win = AstroRackWindow()
        # Verify right panel width
        self.assertEqual(win.right_panel.width(), 350)
        self.assertEqual(win.rack_container.width(), 350)

        # Verify measurement shelf belongs to canvas_container
        self.assertIsNotNone(win.canvas_container)
        self.assertEqual(win.measurement_shelf.parentWidget(), win.canvas_container)

        # Verify shelf minimum size hint has width 0
        self.assertEqual(win.measurement_shelf.minimumSizeHint().width(), 0)

        # Verify scroll area exists in shelf
        self.assertIsNotNone(win.measurement_shelf.scroll_area)

        # Verify collapse button (collapses shelf body, handle stays visible at height 14)
        win.toolbox.toggle_tool("ruler")
        self.assertFalse(win.measurement_shelf.isHidden())
        self.assertFalse(win.measurement_shelf.is_collapsed)
        self.assertEqual(win.measurement_shelf.height(), 60)

        win.measurement_shelf.btn_close.click()
        self.assertTrue(win.measurement_shelf.is_collapsed)
        self.assertTrue(win.measurement_shelf.shelf_body.isHidden())
        self.assertEqual(win.measurement_shelf.height(), 14)
        self.assertFalse(win.measurement_shelf.isHidden())
        win.close()

    def test_bounding_box_drag_and_calibrated_disk(self):
        from PyQt6.QtGui import QMouseEvent, QKeyEvent
        from PyQt6.QtCore import QPointF, Qt

        canvas = AstroCanvas()
        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")

        # Press at (200, 200)
        ev_press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(200.0, 200.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev_press)
        self.assertTrue(canvas.disk_is_dragging)

        # Drag to (500, 200) along diameter -> diameter=300 px, radius=150 px
        ev_move = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(500.0, 200.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseMoveEvent(ev_move)
        self.assertIsNotNone(canvas.temp_disk_rect)
        self.assertEqual(canvas.temp_disk_rect.width(), 300.0)
        self.assertEqual(canvas.disk_radius, 150.0)

        # Release -> pending disk
        ev_rel = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(500.0, 200.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseReleaseEvent(ev_rel)
        self.assertFalse(canvas.disk_is_dragging)
        self.assertIsNotNone(canvas.pending_disk)
        self.assertEqual(canvas.pending_disk[2], 150.0)

        # Apply calibration via Enter key
        ev_enter = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(ev_enter)
        self.assertIsNotNone(canvas.calibrated_disk)
        self.assertEqual(canvas.calibrated_disk[2], 150.0)
        self.assertEqual(canvas._meas_submode, "ruler")
        self.assertIn("150 px", canvas.measurement_hud.lbl_disk_status.text())

    def test_contextual_bottom_shelf_visibility(self):
        hud = MeasurementHUD()

        # 1. Ruler mode (default, uncalibrated - NO 'Off' badge, status box hidden)
        hud.set_mode("ruler")
        self.assertTrue(hud.disk_status_box.isHidden())
        self.assertTrue(hud.results_box.isHidden())
        self.assertTrue(hud.btn_apply_disk.isHidden())
        self.assertTrue(hud.disk_methods_widget.isHidden())
        self.assertTrue(hud.ellipse_step_box.isHidden())

        # 2. Disk mode (uncalibrated - methods visible, status box hidden)
        hud.set_mode("disk")
        self.assertFalse(hud.disk_methods_widget.isHidden())
        self.assertTrue(hud.disk_status_box.isHidden())
        self.assertTrue(hud.btn_apply_disk.isHidden())
        self.assertTrue(hud.btn_cancel_disk.isHidden())
        self.assertTrue(hud.ellipse_step_box.isHidden())
        self.assertTrue(hud.results_box.isHidden())

        # 3. Disk mode (drawing / pending apply active -> [Рамка] [✓] [✕])
        hud.set_disk_prompt("Растяните рамку", can_apply=True)
        self.assertFalse(hud.btn_apply_disk.isHidden())
        self.assertFalse(hud.btn_cancel_disk.isHidden())
        self.assertFalse(hud.btn_method_box.isHidden())
        self.assertTrue(hud.btn_method_3pts.isHidden())

        # 4. Disk mode (calibrated -> methods hidden, compact status badge shown)
        hud.set_disk_status(True, 250.0)
        self.assertTrue(hud.disk_methods_widget.isHidden())
        self.assertFalse(hud.disk_status_box.isHidden())
        self.assertIn("R: 250 px", hud.lbl_disk_status.text())
        self.assertEqual(hud.btn_reset_disk.text(), "✕ Сброс")

        # 5. Ruler mode (calibrated -> compact calibrated badge shown)
        hud.set_mode("ruler")
        self.assertFalse(hud.disk_status_box.isHidden())
        self.assertIn("Calibrated", hud.lbl_disk_status.text())
        self.assertEqual(hud.btn_reset_disk.text(), "✕")

        # 6. Ellipse mode
        hud.set_mode("ellipse")
        self.assertFalse(hud.ellipse_step_box.isHidden())
        self.assertTrue(hud.results_box.isHidden())
        self.assertTrue(hud.disk_methods_widget.isHidden())
        self.assertTrue(hud.disk_status_box.isHidden())
        self.assertTrue(hud.btn_apply_disk.isHidden())

    def test_disk_move_and_resize_transformation(self):
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QPointF, QRectF, Qt

        canvas = AstroCanvas()
        pix = QPixmap(1000, 1000)
        canvas._item.processed_pixmap = pix
        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")

        # Establish initial disk at (500, 500) with R=100
        canvas.disk_center = QPointF(500.0, 500.0)
        canvas._disk_draft_center = canvas.disk_center
        canvas.disk_radius = 100.0
        canvas._disk_draft_radius = 100.0
        canvas.temp_disk_rect = QRectF(400.0, 400.0, 200.0, 200.0)
        canvas.pending_disk = (500.0, 500.0, 100.0)

        # 1. Click inside (>10px from edge, e.g. at (500, 500)) -> Move mode
        c_vp = QPointF(canvas.mapFromScene(QPointF(500.0, 500.0)))
        ev_press_move = QMouseEvent(QMouseEvent.Type.MouseButtonPress, c_vp, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev_press_move)
        self.assertEqual(canvas._disk_transform_mode, "move")

        # Drag center to (530, 520) -> delta = (+30, +20)
        drag_vp = QPointF(canvas.mapFromScene(QPointF(530.0, 520.0)))
        ev_drag_move = QMouseEvent(QMouseEvent.Type.MouseMove, drag_vp, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseMoveEvent(ev_drag_move)
        self.assertEqual(canvas.disk_center.x(), 530.0)
        self.assertEqual(canvas.disk_center.y(), 520.0)
        self.assertEqual(canvas.disk_radius, 100.0)  # Radius unchanged!

        ev_rel_move = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, drag_vp, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseReleaseEvent(ev_rel_move)
        self.assertEqual(canvas.pending_disk[0], 530.0)
        self.assertEqual(canvas.pending_disk[1], 520.0)
        self.assertEqual(canvas.pending_disk[2], 100.0)

        # 2. Click near edge (radius=100 from center (530, 520) -> edge is at (630, 520)) -> Resize mode
        edge_vp = QPointF(canvas.mapFromScene(QPointF(630.0, 520.0)))
        ev_press_resize = QMouseEvent(QMouseEvent.Type.MouseButtonPress, edge_vp, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev_press_resize)
        self.assertEqual(canvas._disk_transform_mode, "resize")

        # Drag to (680, 520) -> radius expands to 150
        drag_res_vp = QPointF(canvas.mapFromScene(QPointF(680.0, 520.0)))
        ev_drag_resize = QMouseEvent(QMouseEvent.Type.MouseMove, drag_res_vp, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseMoveEvent(ev_drag_resize)
        self.assertEqual(canvas.disk_center.x(), 530.0)  # Center unchanged!
        self.assertEqual(canvas.disk_radius, 150.0)      # Radius updated!

        ev_rel_resize = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, drag_res_vp, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseReleaseEvent(ev_rel_resize)
        self.assertEqual(canvas.pending_disk[2], 150.0)

        # 3. Click well outside (e.g. at (100, 100), distance > 150+10) -> resets current rect and starts new box
        out_vp = QPointF(canvas.mapFromScene(QPointF(100.0, 100.0)))
        ev_press_outside = QMouseEvent(QMouseEvent.Type.MouseButtonPress, out_vp, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev_press_outside)
        self.assertEqual(canvas._disk_transform_mode, "draw")
        self.assertEqual(canvas.disk_center, QPointF(100.0, 100.0))
        self.assertEqual(canvas.disk_radius, 0.0)

    def test_three_points_method_sets_pending_disk_and_enables_apply(self):
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QPointF, Qt

        canvas = AstroCanvas()
        pix = QPixmap(1000, 1000)
        canvas._item.processed_pixmap = pix
        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")
        canvas.set_disk_method("three_points")

        # 3 points on circle of radius 100 centered at (500, 500):
        # P1 = (400, 500), P2 = (500, 400), P3 = (600, 500)
        p1 = QPointF(400.0, 500.0)
        p2 = QPointF(500.0, 400.0)
        p3 = QPointF(600.0, 500.0)

        ev1 = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(canvas.mapFromScene(p1)), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev1)
        self.assertEqual(len(canvas.disk_points), 1)

        ev2 = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(canvas.mapFromScene(p2)), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev2)
        self.assertEqual(len(canvas.disk_points), 2)

        ev3 = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(canvas.mapFromScene(p3)), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev3)
        # 3 points are cleared to prevent ghost circle, and unified temp_disk_rect is created
        self.assertEqual(len(canvas.disk_points), 0)
        self.assertEqual(len(canvas.three_points), 0)
        self.assertIsNotNone(canvas.temp_disk_rect)

        # Pending disk should be created and apply button enabled
        self.assertIsNotNone(canvas.pending_disk)
        self.assertAlmostEqual(canvas.pending_disk[0], 500.0, delta=1.0)
        self.assertAlmostEqual(canvas.pending_disk[1], 500.0, delta=1.0)
        self.assertAlmostEqual(canvas.pending_disk[2], 100.0, delta=1.0)
        self.assertTrue(canvas.measurement_hud.btn_apply_disk.isEnabled())

    def test_crater_ellipse_differentiation_and_label_formatting(self):
        # When b < a, True Diameter must be strictly greater than Apparent Diameter
        p_start = (100.0, 100.0)
        p_end = (300.0, 100.0)      # 2a = 200 px, a = 100 px
        p_cur = (200.0, 150.0)      # b = 50 px, cos theta = 0.5, tilt = 60°
        scale_km_per_px = 1.5

        res = calculate_crater_ellipse(p_start, p_end, p_cur, scale_km_per_px)
        self.assertEqual(res["semi_major_a"], 100.0)
        self.assertEqual(res["semi_minor_b"], 50.0)

        true_dia = res["true_diameter_km"]      # 2 * 100 * 1.5 = 300 km
        apparent_dia = res["apparent_diameter_km"]  # 2 * 50 * 1.5 = 150 km
        self.assertEqual(true_dia, 300.0)
        self.assertEqual(apparent_dia, 150.0)
        self.assertGreater(true_dia, apparent_dia)
        self.assertAlmostEqual(res["cos_theta"], 0.5, delta=0.01)
        self.assertAlmostEqual(res["tilt_deg"], 60.0, delta=0.5)

        hud = MeasurementHUD()
        hud.set_ellipse_results(
            true_dia, apparent_dia, res["tilt_deg"], res["cos_theta"],
            res["eccentricity"], 100.0, 50.0
        )
        self.assertIn("True Dia: 300.0 km", hud.row_res_1.text())
        self.assertIn("Apparent: 150.0 km", hud.row_res_2.text())
        self.assertIn("Tilt: 60° (cos θ = 0.50)", hud.row_res_3.text())

    def test_bottom_shelf_compact_icons_and_layout(self):
        hud = MeasurementHUD()
        # Проверяем компактные иконки без громоздкого текста
        self.assertEqual(hud.btn_reset_disk.text(), "✕")
        self.assertEqual(hud.btn_reset_disk.width(), 24)
        self.assertEqual(hud.btn_reset_disk.height(), 24)
        self.assertIn("Сбросить", hud.btn_reset_disk.toolTip())

        self.assertEqual(hud.btn_apply_disk.text(), "✓")
        self.assertEqual(hud.btn_apply_disk.width(), 24)
        self.assertEqual(hud.btn_apply_disk.height(), 24)
        self.assertIn("Применить", hud.btn_apply_disk.toolTip())

        # Кнопка справки [?] справа
        self.assertEqual(hud.btn_help.text(), "?")
        self.assertEqual(hud.btn_help.width(), 24)
        self.assertEqual(hud.btn_help.height(), 24)
        self.assertIn("Справочник", hud.btn_help.toolTip())

        # Блок results_box не включен в layout панели (данные отображаются только в плавающем бейдже на холсте)
        inner_widget = hud.scroll_area.widget()
        inner_layout = inner_widget.layout()
        items = [inner_layout.itemAt(i).widget() for i in range(inner_layout.count()) if inner_layout.itemAt(i).widget()]
        self.assertNotIn(hud.results_box, items)

    def test_circle_from_3_points_analytical_accuracy(self):
        from core.geometry_calc import circle_from_3_points

        # 1. Точный круг с центром (300, 400) и радиусом 250
        cx, cy, r = 300.0, 400.0, 250.0
        p1 = QPointF(cx + r, cy)                     # (550, 400)
        p2 = QPointF(cx, cy + r)                     # (300, 650)
        p3 = QPointF(cx - r * 0.6, cy - r * 0.8)     # (150, 200)

        res = circle_from_3_points(p1, p2, p3)
        self.assertIsNotNone(res)
        center, calc_r = res
        self.assertAlmostEqual(center.x(), cx, places=4)
        self.assertAlmostEqual(center.y(), cy, places=4)
        self.assertAlmostEqual(calc_r, r, places=4)

        # 2. Вырожденные коллинеарные точки
        collinear_res = circle_from_3_points(QPointF(10, 10), QPointF(20, 20), QPointF(30, 30))
        self.assertIsNone(collinear_res)

    def test_image_space_handles_hit_testing_with_zoom(self):
        from PyQt6.QtCore import QPointF, QRectF
        from PyQt6.QtGui import QTransform

        canvas = AstroCanvas()
        pix = QPixmap(1000, 1000)
        canvas._item.processed_pixmap = pix
        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")

        # Задаем диск в координатах изображения: центр (500, 500), R = 100
        canvas.temp_disk_rect = QRectF(400.0, 400.0, 200.0, 200.0)
        canvas.disk_center = QPointF(500.0, 500.0)
        canvas.disk_radius = 100.0

        # В координатах изображения без зума (zoom_factor = 1.0)
        canvas.setTransform(QTransform())
        self.assertEqual(canvas._hit_test_disk_rect(QPointF(400.0, 400.0)), "tl")
        self.assertEqual(canvas._hit_test_disk_rect(QPointF(600.0, 400.0)), "tr")
        self.assertEqual(canvas._hit_test_disk_rect(QPointF(600.0, 600.0)), "br")
        self.assertEqual(canvas._hit_test_disk_rect(QPointF(400.0, 600.0)), "bl")
        self.assertEqual(canvas._hit_test_disk_rect(QPointF(500.0, 500.0)), "center")

        # Проверка динамического порога (tol) при зуме: zoom_factor = 2.0 => tol = 6 px
        canvas.setTransform(QTransform().scale(2.0, 2.0))
        # Точка на расстоянии 5 px от угла (входит в допуск 6 px)
        self.assertEqual(canvas._hit_test_disk_rect(QPointF(405.0, 405.0)), "tl")
        # Точка на расстоянии 8 px от угла (не входит в допуск 6 px)
        self.assertNotEqual(canvas._hit_test_disk_rect(QPointF(408.0, 408.0)), "tl")

    def test_no_nested_frames_and_button_order(self):
        hud = MeasurementHUD()
        # Проверяем структуру функциональных блоков (СТРОГО ОДНА внешняя рамка)
        for box, selector in [
            (hud.mode_container, "QFrame#segmentedBox"),
            (hud.disk_methods_widget, "QFrame#diskMethodsBox"),
            (hud.disk_status_box, "QFrame#diskStatusBox"),
            (hud.ellipse_step_box, "QFrame#ellipseStepBox"),
        ]:
            self.assertIn(selector, box.styleSheet())
            self.assertIn("border: 1px solid #2a3441", box.styleSheet())
            self.assertIn("background-color: #161920", box.styleSheet())

        # Проверяем порядок внутри disk_methods_widget: [btn_method_box, btn_method_3pts, btn_apply_disk, btn_cancel_disk]
        method_layout = hud.disk_methods_widget.layout()
        self.assertEqual(method_layout.count(), 4)
        self.assertEqual(method_layout.itemAt(0).widget(), hud.btn_method_box)
        self.assertEqual(method_layout.itemAt(1).widget(), hud.btn_method_3pts)
        self.assertEqual(method_layout.itemAt(2).widget(), hud.btn_apply_disk)
        self.assertEqual(method_layout.itemAt(3).widget(), hud.btn_cancel_disk)

        # Проверяем порядок внутри disk_status_box: [lbl_disk_status] [btn_reset_disk]
        layout = hud.disk_status_box.layout()
        self.assertEqual(layout.count(), 2)
        self.assertEqual(layout.itemAt(0).widget(), hud.lbl_disk_status)
        self.assertEqual(layout.itemAt(1).widget(), hud.btn_reset_disk)

        # Проверяем отсутствие вложенных рамок у текстовых индикаторов
        self.assertIn("border: none", hud.lbl_disk_status.styleSheet())
        self.assertIn("background: transparent", hud.lbl_disk_status.styleSheet())
        self.assertIn("border: none", hud.lbl_ellipse_step.styleSheet())
        self.assertIn("background: transparent", hud.lbl_ellipse_step.styleSheet())

    def test_no_parasitic_os_window(self):
        hud = MeasurementHUD()
        # В любых режимах results_box обязан оставаться дочерним скрытым виджетом и не создавать отдельное окно OS
        hud.set_mode("ruler")
        self.assertTrue(hud.results_box.isHidden())
        self.assertFalse(hud.results_box.isWindow())

        hud.set_mode("disk")
        self.assertTrue(hud.results_box.isHidden())
        self.assertFalse(hud.results_box.isWindow())

        hud.set_mode("ellipse")
        self.assertTrue(hud.results_box.isHidden())
        self.assertFalse(hud.results_box.isWindow())

    def test_zoom_pan_invariance_and_image_coordinates(self):
        from PyQt6.QtCore import QPointF, QRectF
        from PyQt6.QtGui import QTransform

        canvas = AstroCanvas()
        pix = QPixmap(1000, 1000)
        canvas._item.processed_pixmap = pix
        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")

        # Сохранение в координатах изображения
        canvas.disk_rect_img = QRectF(150.0, 250.0, 300.0, 300.0)
        self.assertEqual(canvas.temp_disk_rect, canvas.disk_rect_img)
        self.assertEqual(canvas.disk_rect_img.center(), QPointF(300.0, 400.0))

        # Панорамирование и масштабирование холста не меняет координаты диска в пространстве изображения
        canvas.setTransform(QTransform().translate(120.0, -80.0).scale(2.5, 2.5))
        self.assertEqual(canvas.disk_rect_img.topLeft(), QPointF(150.0, 250.0))
        self.assertEqual(canvas.disk_rect_img.width(), 300.0)

        # Проверка свойств точек лимба
        canvas.disk_points = [QPointF(100, 100), QPointF(200, 100), QPointF(150, 200)]
        self.assertEqual(canvas.p1_img, QPointF(100, 100))
        self.assertEqual(canvas.p2_img, QPointF(200, 100))
        self.assertEqual(canvas.p3_img, QPointF(150, 200))

    def test_symmetric_resize_from_center(self):
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QPointF, QRectF, Qt

        canvas = AstroCanvas()
        pix = QPixmap(1000, 1000)
        canvas._item.processed_pixmap = pix
        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")

        # Исходный круг: центр (500, 500), R = 100
        canvas.disk_center = QPointF(500.0, 500.0)
        canvas._disk_draft_center = canvas.disk_center
        canvas.disk_radius = 100.0
        canvas._disk_draft_radius = 100.0
        canvas.disk_rect_img = QRectF(400.0, 400.0, 200.0, 200.0)
        canvas.pending_disk = (500.0, 500.0, 100.0)

        # Клик на правом нижнем углу (600, 600) -> режим resize
        br_vp = QPointF(canvas.mapFromScene(QPointF(600.0, 600.0)))
        ev_press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, br_vp, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev_press)
        self.assertEqual(canvas._disk_transform_mode, "resize")

        # Перетаскиваем в точку (650, 620)
        # new_radius = max(abs(650 - 500), abs(620 - 500)) = max(150, 120) = 150
        drag_vp = QPointF(canvas.mapFromScene(QPointF(650.0, 620.0)))
        ev_move = QMouseEvent(QMouseEvent.Type.MouseMove, drag_vp, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseMoveEvent(ev_move)

        # Центр обязан остаться СТРОГО фиксированным на (500, 500)!
        self.assertEqual(canvas.disk_center.x(), 500.0)
        self.assertEqual(canvas.disk_center.y(), 500.0)
        self.assertEqual(canvas.disk_rect_img.center(), QPointF(500.0, 500.0))
        # Радиус стал 150
        self.assertEqual(canvas.disk_radius, 150.0)
        self.assertEqual(canvas.disk_rect_img.width(), 300.0)
        self.assertEqual(canvas.disk_rect_img.height(), 300.0)
        self.assertEqual(canvas.disk_rect_img.topLeft(), QPointF(350.0, 350.0))

    def test_letterbox_screen_and_image_coordinate_conversion(self):
        from PyQt6.QtCore import QPointF, QRectF
        from PyQt6.QtGui import QTransform

        canvas = AstroCanvas()
        # Имитируем letterbox отступ (например, окно 1200x800, а кадр отображается с отступом 100 px слева и 50 px сверху)
        canvas.image_viewport_rect = QRectF(100.0, 50.0, 1000.0, 700.0)

        # 1. При зуме 1.0
        canvas.setTransform(QTransform())
        # Точка экрана (100, 50) должна точно проецироваться в (0, 0) изображения
        self.assertEqual(canvas.screen_to_image(QPointF(100.0, 50.0)), QPointF(0.0, 0.0))
        # Точка изображения (0, 0) переходит в (100, 50) экрана
        self.assertEqual(canvas.image_to_screen(QPointF(0.0, 0.0)), QPointF(100.0, 50.0))
        # Произвольная точка (300, 250) -> screen = 100 + 300 = 400, 50 + 250 = 300
        self.assertEqual(canvas.image_to_screen(QPointF(300.0, 250.0)), QPointF(400.0, 300.0))
        self.assertEqual(canvas.screen_to_image(QPointF(400.0, 300.0)), QPointF(300.0, 250.0))

        # 2. При зуме 2.0
        canvas.setTransform(QTransform().scale(2.0, 2.0))
        # Точка (100, 50) по-прежнему (0, 0)
        self.assertEqual(canvas.screen_to_image(QPointF(100.0, 50.0)), QPointF(0.0, 0.0))
        # Пиксель (10, 20) -> screen = 100 + 10 * 2 = 120, 50 + 20 * 2 = 90
        self.assertEqual(canvas.image_to_screen(QPointF(10.0, 20.0)), QPointF(120.0, 90.0))
        self.assertEqual(canvas.screen_to_image(QPointF(120.0, 90.0)), QPointF(10.0, 20.0))

    def test_unified_move_and_resize_for_three_points(self):
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QPointF, Qt

        canvas = AstroCanvas()
        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")
        canvas.set_disk_method("three_points")

        # 3 точки на окружности с центром (300, 300) и радиусом 100
        canvas.disk_points = [
            QPointF(300.0, 400.0),  # низ (300, 400)
            QPointF(400.0, 300.0),  # право (400, 300)
            QPointF(300.0, 200.0)   # верх (300, 200)
        ]
        # Симулируем 3-й клик через mousePressEvent
        canvas.disk_points = [QPointF(300.0, 400.0), QPointF(400.0, 300.0)]
        ev_p3 = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(300.0, 200.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev_p3)

        # Окружность должна быть рассчитана и перейти в интерактивную temp_disk_rect
        self.assertIsNotNone(canvas.temp_disk_rect)
        self.assertAlmostEqual(canvas.disk_center.x(), 300.0, delta=1.0)
        self.assertAlmostEqual(canvas.disk_center.y(), 300.0, delta=1.0)
        self.assertAlmostEqual(canvas.disk_radius, 100.0, delta=1.0)

        # 1. Проверяем перемещение круга (Move): клик в центре (300, 300)
        ev_center_press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(300.0, 300.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev_center_press)
        self.assertEqual(canvas._disk_transform_mode, "move")

        # Перетаскиваем в (350, 320)
        ev_move = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(350.0, 320.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseMoveEvent(ev_move)
        self.assertAlmostEqual(canvas.disk_center.x(), 350.0, delta=1.0)
        self.assertAlmostEqual(canvas.disk_center.y(), 320.0, delta=1.0)

        # 2. Клик снаружи диска сбрасывает геометрию и начинает построение заново (устанавливает точку 1)
        ev_outside = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(10.0, 10.0), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev_outside)
        self.assertIsNone(canvas.temp_disk_rect)
        self.assertEqual(len(canvas.disk_points), 1)

    def test_contextual_button_switching_phase_a_and_b(self):
        hud = MeasurementHUD()
        hud.set_mode("disk")
        hud.set_disk_method("center_radius")

        # По умолчанию: видны обе кнопки методов, [✓] и [✕] скрыты
        self.assertFalse(hud.btn_method_box.isHidden())
        self.assertFalse(hud.btn_method_3pts.isHidden())
        self.assertTrue(hud.btn_apply_disk.isHidden())
        self.assertTrue(hud.btn_cancel_disk.isHidden())

        # Фаза А (Рамка растягивается): метод 3 точки скрывается, появляются [✓] и [✕]
        hud.set_disk_prompt("Растяните рамку", can_apply=True)
        self.assertFalse(hud.btn_method_box.isHidden())
        self.assertTrue(hud.btn_method_3pts.isHidden())
        self.assertFalse(hud.btn_apply_disk.isHidden())
        self.assertFalse(hud.btn_cancel_disk.isHidden())

        # Фаза Б (Калибровка подтверждена): блок методов скрыт, виден бейдж статуса [● R: 180 px] [✕ Сброс]
        hud.set_disk_status(True, 180.0)
        self.assertTrue(hud.disk_methods_widget.isHidden())
        self.assertFalse(hud.disk_status_box.isHidden())
        self.assertIn("R: 180 px", hud.lbl_disk_status.text())
        self.assertEqual(hud.btn_reset_disk.text(), "✕ Сброс")

        # Переключение на линейку в откалиброванном состоянии: [● Calibrated (R: 180 px)] [✕]
        hud.set_mode("ruler")
        self.assertFalse(hud.disk_status_box.isHidden())
        self.assertIn("Calibrated", hud.lbl_disk_status.text())
        self.assertEqual(hud.btn_reset_disk.text(), "✕")

        # Сброс калибровки: плашка статуса скрывается полностью (нет 'Off' бейджа)
        hud.set_disk_status(False)
        self.assertTrue(hud.disk_status_box.isHidden())

    def test_edge_to_edge_diameter_drag(self):
        import math
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QPointF, Qt

        canvas = AstroCanvas()
        canvas.set_ruler_mode(True)
        canvas.set_measurement_submode("disk")

        # Клик на одном краю лимба (150, 200)
        p_start = QPointF(150.0, 200.0)
        ev_press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, p_start, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mousePressEvent(ev_press)

        # Тянем к противоположному краю лимба (450, 600) -> dx=300, dy=400, hypot=500
        p_cur = QPointF(450.0, 600.0)
        ev_move = QMouseEvent(QMouseEvent.Type.MouseMove, p_cur, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseMoveEvent(ev_move)

        # Центр обязан быть строго серединой отрезка: (150+450)/2 = 300, (200+600)/2 = 400
        self.assertEqual(canvas.disk_center, QPointF(300.0, 400.0))
        # Радиус равен половине длины отрезка: 500 / 2 = 250
        self.assertEqual(canvas.disk_radius, 250.0)
        # Bounding rect построен симметрично вокруг центра: [50, 150, 500, 500]
        self.assertEqual(canvas.temp_disk_rect.topLeft(), QPointF(50.0, 150.0))
        self.assertEqual(canvas.temp_disk_rect.width(), 500.0)
        self.assertEqual(canvas.temp_disk_rect.height(), 500.0)

        # Обе точки (P_start и P_cur) лежат СТРОГО на окружности:
        c = canvas.disk_center
        r = canvas.disk_radius
        dist_start = math.hypot(p_start.x() - c.x(), p_start.y() - c.y())
        dist_cur = math.hypot(p_cur.x() - c.x(), p_cur.y() - c.y())
        self.assertAlmostEqual(dist_start, r, places=5)
        self.assertAlmostEqual(dist_cur, r, places=5)

        # Отпускание мыши фиксирует pending_disk
        ev_rel = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, p_cur, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        canvas.mouseReleaseEvent(ev_rel)
        self.assertEqual(canvas.pending_disk, (300.0, 400.0, 250.0))

    def test_accent_line_state_expanded_and_collapsed(self):
        hud = MeasurementHUD()

        # 1. Развёрнутое состояние (is_expanded == True): тёмная нейтральная граница
        self.assertTrue(hud.is_expanded)
        self.assertFalse(hud.is_collapsed)
        self.assertIn("border-top: 1px solid #1a202c", hud.tab_handle.styleSheet())
        self.assertIn("border-top: 1px solid #1a202c", hud.styleSheet())

        # 2. Свёрнутое состояние (is_expanded == False): неоновая бирюзовая линия 2px
        hud.is_expanded = False
        self.assertTrue(hud.is_collapsed)
        self.assertFalse(hud.is_expanded)
        self.assertIn("border-top: 2px solid #00e5ff", hud.tab_handle.styleSheet())


if __name__ == "__main__":
    unittest.main()
