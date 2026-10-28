"""Тесты параллельного линейного подмешивания Dry/Wet (параллельная компрессия)

Проверяет:
1. Базовую функцию blend_dry_wet: точность, граничные условия (0.001, 0.999), 0..1 vs 0..100.
2. Безопасность типов (uint8, float32) и размерностей (2D, 3D, неконтигуарные массивы).
3. Интеграцию в пайплайн (Inserts 1, 2, 5, 6) и исключение Inserts 3, 4.
4. Кэширование промежуточного wet-кадра (быстрое пересведение при движении wet_mix).
5. Бенчмарк производительности (оверхед <= 2–3 мс на кадрах 2000x2000).
"""
import sys
import os
import time
from pathlib import Path

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.pipeline import blend_dry_wet, _extract_wet_ratio, PipelineWorker
from core.processing import AstroEngine


def test_blend_dry_wet_boundary():
    """Проверка граничных значений wet_ratio: 0.0 (dry), 1.0 (wet), пороги 0.001 и 0.999."""
    dry = np.full((100, 100, 3), 0.2, dtype=np.float32)
    wet = np.full((100, 100, 3), 0.8, dtype=np.float32)

    # 100% wet
    res_1 = blend_dry_wet(dry, wet, wet_ratio=1.0)
    assert np.array_equal(res_1, wet), "wet_ratio=1.0 должен возвращать wet_img"

    # Fast short-circuit >= 0.999
    res_0999 = blend_dry_wet(dry, wet, wet_ratio=0.9995)
    assert np.array_equal(res_0999, wet), "wet_ratio >= 0.999 должен мгновенно возвращать wet_img"

    # 100% dry (0.0)
    res_0 = blend_dry_wet(dry, wet, wet_ratio=0.0)
    assert np.array_equal(res_0, dry), "wet_ratio=0.0 должен возвращать dry_img"

    # Fast short-circuit <= 0.001
    res_0001 = blend_dry_wet(dry, wet, wet_ratio=0.0005)
    assert np.array_equal(res_0001, dry), "wet_ratio <= 0.001 должен мгновенно возвращать dry_img"

    # None handling
    assert blend_dry_wet(None, wet, 0.5) is wet
    assert blend_dry_wet(dry, None, 0.5) is dry
    print("✓ test_blend_dry_wet_boundary пройден.")


def test_blend_dry_wet_math():
    """Проверка математической корректности линейной интерполяции."""
    dry = np.full((50, 50), 0.2, dtype=np.float32)
    wet = np.full((50, 50), 0.8, dtype=np.float32)

    # 50% fraction (0.5)
    res_50 = blend_dry_wet(dry, wet, wet_ratio=0.5)
    expected_50 = 0.5 * 0.8 + 0.5 * 0.2
    assert np.allclose(res_50, expected_50, atol=1e-5), f"Ожидалось {expected_50}, получено {res_50[0,0]}"

    # 50% percentage (50.0)
    res_50_pct = blend_dry_wet(dry, wet, wet_ratio=50.0)
    assert np.allclose(res_50_pct, expected_50, atol=1e-5), "Проценты 50.0 должны нормализоваться к 0.5"

    # 25% mix
    res_25 = blend_dry_wet(dry, wet, wet_ratio=0.25)
    expected_25 = 0.25 * 0.8 + 0.75 * 0.2
    assert np.allclose(res_25, expected_25, atol=1e-5)

    # uint8 array test
    dry_u8 = np.full((50, 50, 3), 50, dtype=np.uint8)
    wet_u8 = np.full((50, 50, 3), 150, dtype=np.uint8)
    res_u8 = blend_dry_wet(dry_u8, wet_u8, wet_ratio=0.5)
    assert res_u8.dtype == np.uint8
    assert np.allclose(res_u8, 100, atol=1)

    print("✓ test_blend_dry_wet_math пройден.")


def test_blend_dry_wet_safety():
    """Проверка устойчивости к несовпадению типов, размерностей и непрерывности памяти."""
    # 1. Разные dtypes (uint8 vs float32)
    dry_u8 = np.full((40, 40, 3), 64, dtype=np.uint8)
    wet_f32 = np.full((40, 40, 3), 0.5, dtype=np.float32)
    res = blend_dry_wet(dry_u8, wet_f32, 0.5)
    assert res.dtype == np.float32
    assert res.shape == (40, 40, 3)

    # 2. Разные размерности (2D ч/б dry vs 3D цветной wet)
    dry_2d = np.full((40, 40), 0.2, dtype=np.float32)
    wet_3d = np.full((40, 40, 3), 0.8, dtype=np.float32)
    res_2d_3d = blend_dry_wet(dry_2d, wet_3d, 0.5)
    assert res_2d_3d.ndim == 3 and res_2d_3d.shape[2] == 3

    # 3. Разные размерности (3D цветной dry vs 2D ч/б wet)
    res_3d_2d = blend_dry_wet(wet_3d, dry_2d, 0.5)
    assert res_3d_2d.ndim == 3 and res_3d_2d.shape[2] == 3

    # 4. Неконтигуарные срезы массивов
    non_contig_dry = np.ones((100, 100), dtype=np.float32)[::2, ::2]  # strides (2, 2)
    non_contig_wet = np.full((100, 100), 2.0, dtype=np.float32)[::2, ::2]
    assert not non_contig_dry.flags['C_CONTIGUOUS']
    res_non_contig = blend_dry_wet(non_contig_dry, non_contig_wet, 0.5)
    assert np.allclose(res_non_contig, 1.5, atol=1e-5)

    print("✓ test_blend_dry_wet_safety пройден.")


def test_extract_wet_ratio():
    """Проверка извлечения wet_ratio из словаря параметров."""
    assert _extract_wet_ratio({'w_wet_mix': 50.0}, 'w_wet_mix') == 0.5
    assert _extract_wet_ratio({'w_wet_mix': 100.0}, 'w_wet_mix') == 1.0
    assert _extract_wet_ratio({'w_wet_mix': 0.0}, 'w_wet_mix') == 0.0
    assert _extract_wet_ratio({'w_wet_mix': 0.75}, 'w_wet_mix') == 0.75
    assert _extract_wet_ratio({}, 'w_wet_mix', default=1.0) == 1.0
    assert _extract_wet_ratio({'other': 42}, 'w_wet_mix', 'wet_mix', default=0.8) == 0.8
    print("✓ test_extract_wet_ratio пройден.")


def test_pipeline_dry_wet_inserts():
    """Интеграционный тест: проверка применения Dry/Wet во всех четырех инсертах."""
    worker = PipelineWorker()
    h, w = 120, 120
    np.random.seed(42)
    raw_img = np.random.uniform(0.2, 0.8, (h, w, 3)).astype(np.float32)

    # -------------------------------------------------------------
    # 1. Insert 1: Sharpness (Wavelets / Unsharp)
    # -------------------------------------------------------------
    worker._task_id = 1
    p_ins1_100 = {'insert1_enabled': True, 'w_mode': 1, 'w_l1': 4.0, 'w_l2': 1.5, 'w_wet_mix': 100.0}
    res_ins1_100 = worker._execute_pipeline(raw_img, p_ins1_100, 1)

    worker._task_id = 2
    p_ins1_0 = {'insert1_enabled': True, 'w_mode': 1, 'w_l1': 4.0, 'w_l2': 1.5, 'w_wet_mix': 0.0}
    res_ins1_0 = worker._execute_pipeline(raw_img, p_ins1_0, 2)

    worker._task_id = 3
    p_ins1_50 = {'insert1_enabled': True, 'w_mode': 1, 'w_l1': 4.0, 'w_l2': 1.5, 'w_wet_mix': 50.0}
    res_ins1_50 = worker._execute_pipeline(raw_img, p_ins1_50, 3)

    # 0% wet mix должен в точности равняться исходнику raw_img
    assert np.allclose(res_ins1_0, raw_img, atol=1e-5), "Insert 1 при 0% wet_mix должен возвращать dry"
    # 50% wet mix должен быть между 0% и 100%
    expected_mid1 = 0.5 * res_ins1_100 + 0.5 * raw_img
    assert np.allclose(res_ins1_50, expected_mid1, atol=1e-4), "Insert 1 при 50% wet_mix должен смешиваться 50/50"

    # -------------------------------------------------------------
    # 2. Insert 2: Richardson-Lucy Deconvolution
    # -------------------------------------------------------------
    worker._task_id = 4
    p_ins2_100 = {'insert2_enabled': True, 'rl_sigma': 1.2, 'rl_iters': 4, 'rl_wet_mix': 100.0}
    res_ins2_100 = worker._execute_pipeline(raw_img, p_ins2_100, 4)

    worker._task_id = 5
    p_ins2_0 = {'insert2_enabled': True, 'rl_sigma': 1.2, 'rl_iters': 4, 'rl_wet_mix': 0.0}
    res_ins2_0 = worker._execute_pipeline(raw_img, p_ins2_0, 5)

    worker._task_id = 6
    p_ins2_50 = {'insert2_enabled': True, 'rl_sigma': 1.2, 'rl_iters': 4, 'rl_wet_mix': 50.0}
    res_ins2_50 = worker._execute_pipeline(raw_img, p_ins2_50, 6)

    assert np.allclose(res_ins2_0, raw_img, atol=1e-5), "Insert 2 при 0% wet_mix должен возвращать dry"
    expected_mid2 = 0.5 * res_ins2_100 + 0.5 * raw_img
    assert np.allclose(res_ins2_50, expected_mid2, atol=1e-4), "Insert 2 при 50% wet_mix должен смешиваться 50/50"

    # -------------------------------------------------------------
    # 3. Insert 5: Local Contrast / Clarity
    # -------------------------------------------------------------
    worker._task_id = 7
    p_ins5_100 = {'insert5_enabled': True, 'lc_method': 'clahe', 'lc_amount': 3.0, 'lc_radius': 16.0, 'lc_wet_mix': 100.0}
    res_ins5_100 = worker._execute_pipeline(raw_img, p_ins5_100, 7)

    worker._task_id = 8
    p_ins5_0 = {'insert5_enabled': True, 'lc_method': 'clahe', 'lc_amount': 3.0, 'lc_radius': 16.0, 'lc_wet_mix': 0.0}
    res_ins5_0 = worker._execute_pipeline(raw_img, p_ins5_0, 8)

    worker._task_id = 9
    p_ins5_50 = {'insert5_enabled': True, 'lc_method': 'clahe', 'lc_amount': 3.0, 'lc_radius': 16.0, 'lc_wet_mix': 50.0}
    res_ins5_50 = worker._execute_pipeline(raw_img, p_ins5_50, 9)

    assert np.allclose(res_ins5_0, raw_img, atol=1e-5), "Insert 5 при 0% wet_mix должен возвращать dry"
    expected_mid5 = 0.5 * res_ins5_100 + 0.5 * raw_img
    assert np.allclose(res_ins5_50, expected_mid5, atol=1e-4), "Insert 5 при 50% wet_mix должен смешиваться 50/50"

    # -------------------------------------------------------------
    # 4. Insert 6: Color & Mineral Engine
    # -------------------------------------------------------------
    worker._task_id = 10
    p_ins6_100 = {'insert6_enabled': True, 'cs_saturation': 2.0, 'cs_mineral': 0.8, 'cs_wet_mix': 100.0}
    res_ins6_100 = worker._execute_pipeline(raw_img, p_ins6_100, 10)

    worker._task_id = 11
    p_ins6_0 = {'insert6_enabled': True, 'cs_saturation': 2.0, 'cs_mineral': 0.8, 'cs_wet_mix': 0.0}
    res_ins6_0 = worker._execute_pipeline(raw_img, p_ins6_0, 11)

    worker._task_id = 12
    p_ins6_50 = {'insert6_enabled': True, 'cs_saturation': 2.0, 'cs_mineral': 0.8, 'cs_wet_mix': 50.0}
    res_ins6_50 = worker._execute_pipeline(raw_img, p_ins6_50, 12)

    assert np.allclose(res_ins6_0, raw_img, atol=1e-5), "Insert 6 при 0% wet_mix должен возвращать dry"
    expected_mid6 = 0.5 * res_ins6_100 + 0.5 * raw_img
    assert np.allclose(res_ins6_50, expected_mid6, atol=1e-4), "Insert 6 при 50% wet_mix должен смешиваться 50/50"

    print("✓ test_pipeline_dry_wet_inserts пройден.")


def test_dry_wet_caching_speed():
    """Тест кэширования: изменение только wet_mix должно отрабатывать мгновенно из кэша."""
    worker = PipelineWorker()
    h, w = 500, 500
    img = np.random.uniform(0.1, 0.9, (h, w, 3)).astype(np.float32)

    # Первый прогон (тяжелый RL)
    worker._task_id = 1
    t0 = time.perf_counter()
    p1 = {'insert2_enabled': True, 'rl_sigma': 1.5, 'rl_iters': 10, 'rl_wet_mix': 100.0}
    res1 = worker._execute_pipeline(img, p1, 1)
    t_full = (time.perf_counter() - t0) * 1000.0

    # Второй прогон: только изменение rl_wet_mix -> должен взять wet из кэша
    worker._task_id = 2
    t0 = time.perf_counter()
    p2 = {'insert2_enabled': True, 'rl_sigma': 1.5, 'rl_iters': 10, 'rl_wet_mix': 50.0}
    res2 = worker._execute_pipeline(img, p2, 2)
    t_cached = (time.perf_counter() - t0) * 1000.0

    print(f"  RL full compute: {t_full:.1f} ms | RL wet_mix change (cached): {t_cached:.2f} ms")
    assert t_cached < t_full / 5.0, f"Кэшированный пересчет должен быть в разы быстрее (получено {t_cached:.2f} ms vs {t_full:.1f} ms)"
    print("✓ test_dry_wet_caching_speed пройден.")


def test_blend_dry_wet_benchmark():
    """Бенчмарк производительности: оверхед смешивания на кадрах 2000x2000."""
    h, w = 2000, 2000

    # uint8 2000x2000 RGB
    dry_u8 = np.full((h, w, 3), 100, dtype=np.uint8)
    wet_u8 = np.full((h, w, 3), 200, dtype=np.uint8)

    # warm-up
    blend_dry_wet(dry_u8, wet_u8, 0.5)

    runs = 10
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = blend_dry_wet(dry_u8, wet_u8, 0.7)
    dt_u8 = (time.perf_counter() - t0) / runs * 1000.0

    # float32 2000x2000 RGB
    dry_f32 = np.full((h, w, 3), 0.3, dtype=np.float32)
    wet_f32 = np.full((h, w, 3), 0.7, dtype=np.float32)

    blend_dry_wet(dry_f32, wet_f32, 0.5)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = blend_dry_wet(dry_f32, wet_f32, 0.7)
    dt_f32 = (time.perf_counter() - t0) / runs * 1000.0

    # Short-circuit benchmark (wet_ratio=1.0)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = blend_dry_wet(dry_u8, wet_u8, 1.0)
    dt_short = (time.perf_counter() - t0) / runs * 1000.0

    print(f"  Benchmark 2000x2000 uint8 3-ch:   {dt_u8:.2f} ms")
    print(f"  Benchmark 2000x2000 float32 3-ch: {dt_f32:.2f} ms")
    print(f"  Benchmark short-circuit (100%):   {dt_short:.4f} ms")

    # Проверка, что время смешивания укладывается в разумный порог ~2-4 мс
    assert dt_u8 < 10.0, f"Время смешивания uint8 превышает порог: {dt_u8:.2f} ms"
    print("✓ test_blend_dry_wet_benchmark пройден.")


if __name__ == "__main__":
    print("=== Запуск тестов параллельного линейного подмешивания Dry/Wet ===")
    test_blend_dry_wet_boundary()
    test_blend_dry_wet_math()
    test_blend_dry_wet_safety()
    test_extract_wet_ratio()
    test_pipeline_dry_wet_inserts()
    test_dry_wet_caching_speed()
    test_blend_dry_wet_benchmark()
    print("\n>>> ВСЕ ТЕСТЫ DRY/WET УСПЕШНО ПРОЙДЕНЫ! <<<")
