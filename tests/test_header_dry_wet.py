"""Тесты регуляторов Dry/Wet в шапках инсертов (Header Dry/Wet controls)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from ui.widgets import (
    WaveletInsert, RichardsonLucyInsert, ToneContrastInsert,
    ColorMonoInsert, LocalContrastInsert, ColorEngineInsert
)
from ui.main_window import AstroRackWindow

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


def test_dry_wet_presence_and_absence():
    ins1 = WaveletInsert()
    ins2 = RichardsonLucyInsert()
    ins3 = ToneContrastInsert()
    ins4 = ColorMonoInsert()
    ins5 = LocalContrastInsert()
    ins6 = ColorEngineInsert()

    # 1. Проверяем наличие в шапках 1, 2, 5, 6
    assert ins1.header.dry_wet_ctrl is not None, "INSERT 1 должен иметь Dry/Wet в шапке"
    assert ins2.header.dry_wet_ctrl is not None, "INSERT 2 должен иметь Dry/Wet в шапке"
    assert ins5.header.dry_wet_ctrl is not None, "INSERT 5 должен иметь Dry/Wet в шапке"
    assert ins6.header.dry_wet_ctrl is not None, "INSERT 6 должен иметь Dry/Wet в шапке"

    # 2. Проверяем отсутствие в шапках 3 и 4
    assert ins3.header.dry_wet_ctrl is None, "INSERT 3 не должен иметь Dry/Wet в шапке"
    assert ins4.header.dry_wet_ctrl is None, "INSERT 4 не должен иметь Dry/Wet в шапке"

    print("PASS test_dry_wet_presence_and_absence")


def test_dry_wet_controls_and_reset():
    ins1 = WaveletInsert()
    ins1.set_enabled(True)
    ctrl = ins1.header.dry_wet_ctrl

    assert ctrl.value() == 100
    assert ctrl.ratio() == 1.0
    assert ctrl.lbl_value.text() == "Wet: 100%"

    # Изменение значения
    ctrl.set_value(65)
    assert ctrl.value() == 65
    assert abs(ctrl.ratio() - 0.65) < 1e-4
    assert ctrl.lbl_value.text() == "Wet: 65%"

    # Сброс через reset_default
    ctrl.reset_default()
    assert ctrl.value() == 100
    assert ctrl.lbl_value.text() == "Wet: 100%"

    # Сброс через кнопку R
    ctrl.set_value(30)
    assert ctrl.value() == 30
    ctrl.btn_reset.click()
    assert ctrl.value() == 100

    print("PASS test_dry_wet_controls_and_reset")


def test_main_window_update_insert_param():
    win = AstroRackWindow()

    # Проверяем вызов update_insert_param
    win.update_insert_param(1, 'wet_mix', 0.40)
    assert abs(win.insert1_wavelet.get_dry_wet_ratio() - 0.40) < 1e-3

    win.update_insert_param(2, 'wet_mix', 0.75)
    assert abs(win.insert2_rl.get_dry_wet_ratio() - 0.75) < 1e-3

    win.update_insert_param(5, 'wet_mix', 0.50)
    assert abs(win.insert5_lc.get_dry_wet_ratio() - 0.50) < 1e-3

    win.update_insert_param(6, 'wet_mix', 0.85)
    assert abs(win.insert6_color.get_dry_wet_ratio() - 0.85) < 1e-3

    # Проверяем сброс всех в flat
    win._reset_all_to_flat()
    assert win.insert1_wavelet.get_dry_wet_ratio() == 1.0
    assert win.insert2_rl.get_dry_wet_ratio() == 1.0
    assert win.insert5_lc.get_dry_wet_ratio() == 1.0
    assert win.insert6_color.get_dry_wet_ratio() == 1.0

    win.close()
    print("PASS test_main_window_update_insert_param")


if __name__ == '__main__':
    test_dry_wet_presence_and_absence()
    test_dry_wet_controls_and_reset()
    test_main_window_update_insert_param()
    print("ALL HEADER DRY/WET TESTS PASSED!")
