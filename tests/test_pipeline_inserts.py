"""Интеграционный тест Insert 5 / Insert 6 в PipelineWorker (без GUI-цикла).

Запуск:  python tests/test_pipeline_inserts.py   (или pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pipeline import PipelineWorker  # noqa: E402
from test_enhance import make_moon  # noqa: E402


def _run(worker, params):
    return worker._execute_pipeline(worker._raw_image, params, worker._task_id)


def _worker(img):
    w = PipelineWorker()
    w.set_source_image(img)
    return w


BASE = {'insert1_enabled': False, 'insert2_enabled': False, 'insert3_enabled': False, 'insert4_enabled': False}


def test_inserts_disabled_are_noop():
    img = make_moon(512)
    w = _worker(img)
    out = _run(w, dict(BASE, insert5_enabled=False, insert6_enabled=False))
    assert np.array_equal(out, img)


def test_insert3_tone_applied_and_cached():
    img = make_moon(256)
    w = _worker(img)
    p3 = dict(BASE, insert3_enabled=True, tc_black=0.05, tc_gamma=1.2, tc_contrast=1.1)
    o3 = _run(w, p3)
    assert o3.shape == img.shape and not np.array_equal(o3, img)
    # Повторный запрос — выдача из LRU-кэша стадии 3
    assert _run(w, p3) is o3
    # Проверка экстремальных значений (black >= 1.0, gamma <= 0, contrast <= 0, nan)
    p3_edge = dict(BASE, insert3_enabled=True, tc_black=1.5, tc_gamma=0.0, tc_contrast=0.0)
    o3_edge = _run(w, p3_edge)
    assert o3_edge.shape == img.shape and not np.isnan(o3_edge).any()


def test_insert5_and_insert6_applied_and_cached():
    img = make_moon(512)
    w = _worker(img)
    p5 = dict(BASE, insert5_enabled=True, lc_method='clahe', lc_amount=3.0, lc_radius=16)
    o5 = _run(w, p5)
    assert o5.shape == img.shape and not np.array_equal(o5, img)

    # Повторный запрос с теми же параметрами — результат из кэша (тот же объект)
    assert _run(w, p5) is o5

    p56 = dict(p5, insert6_enabled=True, cs_saturation=1.5, cs_mineral=0.7, cs_bg_neutralize=True)
    o56 = _run(w, p56)
    assert not np.array_equal(o56, o5)
    assert _run(w, p56) is o56

    # Байпас Insert 6 возвращает кадр после Insert 5 без пересчёта; возврат — из кэша стадии 6
    assert _run(w, dict(p56, insert6_enabled=False)) is o5
    assert _run(w, p56) is o56

    # high_pass метод
    ohp = _run(w, dict(p5, lc_method='high_pass', lc_amount=1.0, lc_radius=30))
    assert ohp.shape == img.shape and not np.array_equal(ohp, o5)


def test_insert6_gray_after_green_extract_is_safe():
    img = make_moon(256)
    w = _worker(img)
    params = dict(BASE, insert4_enabled=True, extract_green=True,
                  insert5_enabled=True, lc_method='clahe', lc_amount=2.0, lc_radius=16,
                  insert6_enabled=True, cs_saturation=2.0, cs_mineral=1.0, cs_bg_neutralize=True)
    out = _run(w, params)
    assert out.ndim == 2 and out.shape == img.shape[:2]


def test_master_bypass_returns_raw():
    img = make_moon(256)
    w = _worker(img)
    out = _run(w, dict(BASE, global_bypass=True, insert5_enabled=True, lc_amount=4.0, insert6_enabled=True, cs_saturation=2.0))
    assert np.array_equal(out, img)


if __name__ == "__main__":
    import traceback
    failures = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except Exception:
                failures += 1
                print(f"FAIL {_name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
