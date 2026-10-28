"""Тест UI виджетов Insert 5 (Local Contrast) и Insert 6 (Color Engine) и интеграции в AstroRackWindow."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from ui.widgets import LocalContrastInsert, ColorEngineInsert
from ui.main_window import AstroRackWindow

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


def test_insert5_widget():
    ins5 = LocalContrastInsert()
    assert ins5.header.lbl_title.text() == "INSERT 5: Local Contrast / Clarity"
    assert not ins5.is_enabled()

    vals = ins5.get_values()
    assert vals['lc_method'] == 'clahe'
    assert abs(vals['lc_amount'] - 2.0) < 1e-4
    assert abs(vals['lc_radius'] - 16.0) < 1e-4

    # Переключение в High-Pass
    ins5.combo_mode.setCurrentIndex(1)
    vals_hp = ins5.get_values()
    assert vals_hp['lc_method'] == 'high_pass'
    assert abs(vals_hp['lc_amount'] - 0.8) < 1e-4
    assert abs(vals_hp['lc_radius'] - 30.0) < 1e-4

    # Проверка set_values
    ins5.set_values({'lc_method': 'clahe', 'lc_amount': 3.5, 'lc_radius': 32.0})
    vals_set = ins5.get_values()
    assert vals_set['lc_method'] == 'clahe'
    assert abs(vals_set['lc_amount'] - 3.5) < 1e-4
    assert abs(vals_set['lc_radius'] - 32.0) < 1e-4
    print("PASS test_insert5_widget")


def test_insert6_widget():
    ins6 = ColorEngineInsert()
    assert ins6.header.lbl_title.text() == "INSERT 6: Color Engine & Background"
    assert not ins6.is_enabled()

    vals = ins6.get_values()
    assert abs(vals['cs_saturation'] - 1.0) < 1e-4
    assert abs(vals['cs_mineral'] - 0.0) < 1e-4
    assert vals['cs_bg_neutralize'] is False
    assert vals['cs_bg_coords'] is None

    # Проверка set_values
    ins6.set_values({
        'cs_saturation': 1.8,
        'cs_mineral': 0.75,
        'cs_bg_neutralize': True,
        'cs_bg_coords': (10, 20, 50, 50)
    })
    vals_set = ins6.get_values()
    assert abs(vals_set['cs_saturation'] - 1.8) < 1e-4
    assert abs(vals_set['cs_mineral'] - 0.75) < 1e-4
    assert vals_set['cs_bg_neutralize'] is True
    assert vals_set['cs_bg_coords'] == (10, 20, 50, 50)
    print("PASS test_insert6_widget")


def test_main_window_rack_integration():
    win = AstroRackWindow()
    # Проверяем, что инсерты добавлены в окно
    assert hasattr(win, 'insert5_lc')
    assert hasattr(win, 'insert6_color')
    assert win.insert5_lc is not None
    assert win.insert6_color is not None

    # Проверяем сбор параметров
    state_init = win._get_current_state_dict()
    assert 'insert5_enabled' in state_init
    assert 'insert6_enabled' in state_init
    assert 'lc_method' in state_init
    assert 'cs_saturation' in state_init

    # Проверяем смену пресета
    preset_data = win._builtin_presets["Moon: Mineral Color & Clarity"]
    win._apply_preset_dict(preset_data)

    state = win._get_current_state_dict()
    assert state['insert5_enabled'] is True
    assert state['insert6_enabled'] is True
    assert state['lc_method'] == 'clahe'
    assert abs(state['cs_saturation'] - 1.6) < 1e-4
    assert abs(state['cs_mineral'] - 0.85) < 1e-4
    assert state['cs_bg_neutralize'] is True

    # Проверяем сброс в flat
    win._reset_all_to_flat()
    state_flat = win._get_current_state_dict()
    assert state_flat['insert5_enabled'] is False
    assert state_flat['insert6_enabled'] is False

    win.close()
    print("PASS test_main_window_rack_integration")


if __name__ == '__main__':
    test_insert5_widget()
    test_insert6_widget()
    test_main_window_rack_integration()
    print("ALL UI INSERT TESTS PASSED")
