"""Тесты и бенчмарк Insert 5 (Local Contrast) и Insert 6 (Color Engine).

Запуск:  python -m pytest tests/test_enhance.py -s     или     python tests/test_enhance.py
Лимит времени на кадр 2000x2000 задаётся переменной окружения ASTRO_PERF_MS (по умолчанию 100 мс).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhance import apply_local_contrast, apply_color_engine  # noqa: E402

PERF_LIMIT_MS = float(os.environ.get("ASTRO_PERF_MS", "100"))
N = 2000


def make_moon(n: int = N, seed: int = 0) -> np.ndarray:
    """Синтетический диск «Луны» с кратерами, цветовым оттенком и цветным дрейфом фона."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    c, rad = n / 2.0, n * 0.4
    rr = np.hypot(xx - c, yy - c)
    disc = np.clip((rad - rr) / 4.0, 0.0, 1.0)
    base = 0.15 + 0.7 * (1.0 - (rr / rad) ** 2).clip(0, 1) ** 0.5
    tex = rng.normal(0.0, 0.04, (n, n)).astype(np.float32)
    lum = (base + tex) * disc
    img = np.empty((n, n, 3), np.float32)
    img[..., 0] = lum * 1.05
    img[..., 1] = lum * 0.98
    img[..., 2] = lum * 0.90
    # цветной дрейф фона (R чуть выше, B чуть ниже) + шум
    img += np.array([0.040, 0.025, 0.012], np.float32) * (1.0 - disc)[..., None]
    img += rng.normal(0.0, 0.004, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _time_ms(fn, repeats: int = 5) -> float:
    fn()  # прогрев (аллокации, потоки OpenCV)
    best = 1e9
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


IMG = make_moon()


# ------------------------------- Local Contrast ------------------------------

def test_clahe_increases_local_contrast_and_protects_ends():
    out = apply_local_contrast(IMG, 3.0, 16, 'clahe')
    assert out.dtype == np.float32 and out.shape == IMG.shape
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6
    disc = IMG.mean(axis=2) > 0.2
    assert out[disc].std() != IMG[disc].std()
    # чёрный фон и вход не изменены (защита теней / нет мутации входа)
    bg = IMG.mean(axis=2) < 0.02
    assert np.abs(out[bg] - IMG[bg]).mean() < 0.01


def test_clahe_neutral_at_clip_1():
    out = apply_local_contrast(IMG, 1.0, 16, 'clahe')
    assert np.array_equal(out, IMG) and out is not IMG


def test_high_pass_formula_and_neutral():
    gray = IMG[..., 1]
    assert np.array_equal(apply_local_contrast(gray, 0.0, 30, 'high_pass'), gray)
    out = apply_local_contrast(gray, 1.0, 30, 'high_pass')
    assert out.shape == gray.shape and 0.0 <= out.min() and out.max() <= 1.0
    assert np.abs(out - gray).max() > 1e-3


def test_local_contrast_does_not_mutate_input():
    src = IMG.copy()
    for m, a, r in (('clahe', 3.0, 16), ('high_pass', 1.5, 40)):
        apply_local_contrast(src, a, r, m)
    assert np.array_equal(src, IMG)


def test_local_contrast_dtypes_and_gray():
    u8 = (IMG[:512, :512] * 255).astype(np.uint8)
    for m in ('clahe', 'high_pass'):
        o8 = apply_local_contrast(u8, 2.5, 20, m)
        assert o8.dtype == np.uint8 and o8.shape == u8.shape
        o_g = apply_local_contrast(u8[..., 1], 2.5, 20, m)
        assert o_g.dtype == np.uint8 and o_g.ndim == 2


# ------------------------------- Color Engine --------------------------------

def test_color_engine_gray_passthrough():
    gray = IMG[..., 0]
    assert apply_color_engine(gray, 2.0, 1.0, True) is gray
    g1 = gray[..., None]
    assert apply_color_engine(g1, 2.0, 1.0, True) is g1


def test_saturation_preserves_luma_and_limits_gamut():
    out = apply_color_engine(IMG, saturation=2.0)
    luma = lambda a: 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    dl = np.abs(luma(out) - luma(IMG))
    bright = luma(IMG) > 0.1
    assert dl[bright].max() < 2e-3          # яркость диска сохранена
    assert dl.max() < 1e-2                  # в шумовых тенях отрицательные значения лишь обнуляются
    assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6
    chroma = lambda a: (a.max(axis=2) - a.min(axis=2)).mean()
    assert chroma(out) > chroma(IMG)
    assert np.allclose(apply_color_engine(IMG, 0.0)[..., 0], apply_color_engine(IMG, 0.0)[..., 2], atol=2e-3)


def test_bg_neutralization_auto_and_manual():
    bg_mean = lambda a, sl: a[sl].reshape(-1, 3).mean(axis=0)
    corner = (slice(0, 150), slice(0, 150))
    before = bg_mean(IMG, corner)
    spread_before = before.max() - before.min()
    assert spread_before > 0.02
    for kwargs in ({}, {'bg_coords': (0, 0, 150, 150)}, {'bg_coords': (75, 75)}):
        out = apply_color_engine(IMG, bg_neutralize=True, **kwargs)
        after = bg_mean(out, corner)
        assert after.max() - after.min() < 0.25 * spread_before, (kwargs, before, after)
        assert out.min() >= 0.0
    # диск (терминатор/детали) не срезан: яркие области практически не меняются
    out = apply_color_engine(IMG, bg_neutralize=True)
    disc = IMG.mean(axis=2) > 0.3
    assert np.abs(out[disc] - IMG[disc]).mean() < 0.03


def test_mineral_boost_blue_orange_and_green_suppression():
    px = np.zeros((4, 4, 3), np.float32)
    px[:] = (0.50, 0.50, 0.50)
    px[0, 0] = (0.40, 0.50, 0.60)   # титан (синеватый)
    px[0, 1] = (0.60, 0.50, 0.40)   # оксиды железа (оранжевый)
    px[0, 2] = (0.45, 0.60, 0.45)   # паразитный зелёный
    out = apply_color_engine(px, mineral_boost=1.0)
    assert (out[0, 0, 2] - out[0, 0, 0]) > 0.2 * 1.0           # синий усилен
    assert (out[0, 1, 0] - out[0, 1, 2]) > 0.2 * 1.0           # оранжевый усилен
    assert (out[0, 2, 1] - 0.5 * (out[0, 2, 0] + out[0, 2, 2])) < 0.15 * 0.8  # зелёный подавлен
    assert np.allclose(out[1:], 0.5, atol=1e-4)                # нейтральные серые не окрашены


def test_color_engine_uint8_and_no_mutation():
    src = IMG.copy()
    u8 = (IMG[:512, :512] * 255).astype(np.uint8)
    o = apply_color_engine(u8, 1.5, 0.6, True)
    assert o.dtype == np.uint8 and o.shape == u8.shape
    apply_color_engine(src, 1.5, 0.6, True)
    assert np.array_equal(src, IMG)


# ----------------------------------- Perf ------------------------------------

def test_performance_2000x2000():
    cases = {
        "CLAHE (amount=3, grid=16)": lambda: apply_local_contrast(IMG, 3.0, 16, 'clahe'),
        "CLAHE (amount=5, grid=64)": lambda: apply_local_contrast(IMG, 5.0, 64, 'clahe'),
        "High-pass (sigma=15)": lambda: apply_local_contrast(IMG, 1.0, 15, 'high_pass'),
        "High-pass (sigma=60)": lambda: apply_local_contrast(IMG, 1.0, 60, 'high_pass'),
        "Color: saturation": lambda: apply_color_engine(IMG, 1.6),
        "Color: sat+mineral": lambda: apply_color_engine(IMG, 1.4, 0.8),
        "Color: sat+mineral+bg (auto)": lambda: apply_color_engine(IMG, 1.4, 0.8, True),
        "Color: sat+mineral+bg (rect)": lambda: apply_color_engine(IMG, 1.4, 0.8, True, (0, 0, 200, 200)),
    }
    gray = IMG[..., 1].copy()
    cases["CLAHE gray"] = lambda: apply_local_contrast(gray, 3.0, 16, 'clahe')
    cases["High-pass gray"] = lambda: apply_local_contrast(gray, 1.0, 30, 'high_pass')

    print(f"\n--- 2000x2000 float32, лимит {PERF_LIMIT_MS:.0f} мс ---")
    failed = []
    for name, fn in cases.items():
        ms = _time_ms(fn)
        flag = "OK " if ms <= PERF_LIMIT_MS else "SLOW"
        print(f"[{flag}] {name:<34} {ms:7.1f} мс")
        if ms > PERF_LIMIT_MS:
            failed.append((name, ms))
    assert not failed, f"Превышен лимит {PERF_LIMIT_MS} мс: {failed}"


if __name__ == "__main__":
    try:
        import pytest
    except ImportError:  # запуск без pytest: простой раннер
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
    else:
        sys.exit(pytest.main([__file__, "-s", "-q"]))
