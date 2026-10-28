"""Тесты UI/UX рефакторинга: RotaryKnob, выравнивание слайдеров, суб-байпасы Insert 6, Flyout-меню и тулбар."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint
from ui.widgets import (
    RotaryKnob, PlanetarySlider, ColorEngineInsert,
    ToolboxPanel, PhotoshopToolButton
)
from ui.main_window import AstroRackWindow

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


def test_rotary_knob():
    knob = RotaryKnob(default_val=100)
    assert knob.width() == 24
    assert knob.height() == 24
    assert knob.value() == 100
    assert knob.ratio() == 1.0

    # Проверка изменения значения
    knob.set_value(50)
    assert knob.value() == 50
    assert knob.ratio() == 0.5
    assert "Dry/Wet Mix: 50%" in knob.toolTip()

    # Сброс двойным кликом / reset_default
    knob.reset_default()
    assert knob.value() == 100
    assert "Dry/Wet Mix: 100%" in knob.toolTip()
    print("PASS test_rotary_knob")


def test_planetary_slider_alignment():
    slider = PlanetarySlider(Qt.Orientation.Horizontal)
    slider.resize(200, 24)
    # Клик в самом начале x = 0 должен давать 0
    slider._update_from_mouse(0)
    assert slider.value() == slider.minimum()
    # Клик в конце x = 200 должен давать maximum
    slider._update_from_mouse(200)
    assert slider.value() == slider.maximum()
    print("PASS test_planetary_slider_alignment")


def test_insert6_sub_bypasses():
    ins6 = ColorEngineInsert()
    # По умолчанию суб-байпасы должны присутствовать
    assert ins6.row_saturation.has_sub_bypass is True
    assert ins6.row_mineral.has_sub_bypass is True
    assert ins6.chk_bg_neutralize is not None

    # По умолчанию параметры выключены (bypassed) в flat состоянии
    ins6.set_flat()
    assert ins6.row_saturation.is_param_enabled() is False
    assert ins6.row_mineral.is_param_enabled() is False
    assert ins6.chk_bg_neutralize.isChecked() is False

    # Эффективные значения при выключенном суб-байпасе должны быть нейтральными
    vals = ins6.get_values()
    assert vals['cs_saturation'] == 1.0
    assert vals['cs_mineral'] == 0.0
    assert vals['cs_bg_neutralize'] is False

    # Включение параметров
    ins6.row_saturation.set_param_enabled(True)
    ins6.row_saturation.set_value(1.80)
    ins6.row_mineral.set_param_enabled(True)
    ins6.row_mineral.set_value(0.75)
    ins6.chk_bg_neutralize.setChecked(True)

    vals2 = ins6.get_values()
    assert abs(vals2['cs_saturation'] - 1.80) < 1e-3
    assert abs(vals2['cs_mineral'] - 0.75) < 1e-3
    assert vals2['cs_bg_neutralize'] is True
    assert vals2['cs_saturation_on'] is True
    assert vals2['cs_mineral_on'] is True
    print("PASS test_insert6_sub_bypasses")


def test_photoshop_tool_flyouts():
    toolbox = ToolboxPanel()
    assert isinstance(toolbox.btn_loupe, PhotoshopToolButton)
    assert toolbox.btn_loupe.has_flyout is True
    assert isinstance(toolbox.btn_ruler, PhotoshopToolButton)
    assert toolbox.btn_ruler.has_flyout is True

    # Проверка смены зума лупы
    zoom_received = []
    toolbox.loupe_zoom_changed.connect(lambda f: zoom_received.append(f))
    toolbox._set_loupe_zoom(4.0)
    assert zoom_received == [4.0]

    # Проверка смены единиц и оптики линейки
    unit_received = []
    toolbox.ruler_unit_changed.connect(lambda u: unit_received.append(u))
    toolbox._set_ruler_unit("km")
    assert unit_received == ["km"]

    optics_received = []
    toolbox.ruler_optics_changed.connect(lambda f, p: optics_received.append((f, p)))
    toolbox._set_focal_len(1200.0)
    toolbox._set_pixel_size(2.90)
    assert (1200.0, 3.76) in optics_received
    assert (1200.0, 2.90) in optics_received
    print("PASS test_photoshop_tool_flyouts")


def test_main_window_alignment():
    win = AstroRackWindow()
    win.show()
    import numpy as np
    win._raw_image = np.zeros((100, 100, 3), dtype=np.float32)
    win._update_canvas_guard_state()
    # Проверка наличия toolbox и canvas
    assert win.toolbox is not None
    assert win.canvas is not None

    # Проверка высоты шапок инсертов (строго 40 px с симметричными отступами 8px)
    assert win.insert1_wavelet.header.height() == 40
    assert win.insert2_rl.header.height() == 40
    assert win.insert3_tone.header.height() == 40
    assert win.insert4_mono.header.height() == 40
    assert win.insert5_lc.header.height() == 40
    assert win.insert6_color.header.height() == 40

    # Проверка компактности полей ввода (44 px)
    assert win.insert5_lc.row_amount.spinbox.width() == 44

    # Проверка кнопки горячих клавиш
    assert win.btn_hotkeys.text() == "⌨"

    # Проверка тумблера State A / B и динамической смены темы
    assert win.btn_state_toggle.text() == "State: [ A ]"
    assert win.insert1_wavelet.header._theme['name'] == 'A'
    assert win.insert5_lc.row_amount.slider._theme['name'] == 'A'

    win.btn_state_toggle.click()
    assert win.btn_state_toggle.text() == "State: [ B ]"
    assert win.insert1_wavelet.header._theme['name'] == 'B'
    assert win.insert5_lc.row_amount.slider._theme['name'] == 'B'

    win.btn_state_toggle.click()
    assert win.btn_state_toggle.text() == "State: [ A ]"
    assert win.insert1_wavelet.header._theme['name'] == 'A'
    assert win.insert5_lc.row_amount.slider._theme['name'] == 'A'

    # Проверка подключения единиц линейки к канвасу
    win.toolbox._set_ruler_unit("km")
    assert win.canvas._ruler_unit == "km"

    win.toolbox._set_focal_len(1500.0)
    assert win.canvas._telescope_focal_len == 1500.0
    win.close()
    print("PASS test_main_window_alignment")


if __name__ == '__main__':
    test_rotary_knob()
    test_planetary_slider_alignment()
    test_insert6_sub_bypasses()
    test_photoshop_tool_flyouts()
    test_main_window_alignment()
    print("ALL UI REFACTOR TESTS PASSED!")
