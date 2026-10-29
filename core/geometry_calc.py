"""
Модуль геометрических расчётов для астрономических измерений в AstroRack DAW.
Включает аналитический расчёт окружности по 3 точкам лимба,
компенсацию перспективного сферического сжатия (ракурса Луны cos θ)
и аналитику эллиптического замера кратеров.
"""

import math
from typing import Optional, Tuple, Dict, Any


def circle_from_3_points(p1, p2, p3):
    """
    Строгий аналитический расчёт центра и радиуса окружности по 3 точкам в координатах изображения.
    Использует точку пересечения серединных перпендикуляров.

    Возвращает:
        (center: QPointF, radius: float) если точки образуют невырожденный треугольник,
        None если точки коллинеарны или радиус вне допустимого диапазона.
    """
    x1 = p1.x() if hasattr(p1, 'x') else float(p1[0])
    y1 = p1.y() if hasattr(p1, 'y') else float(p1[1])
    x2 = p2.x() if hasattr(p2, 'x') else float(p2[0])
    y2 = p2.y() if hasattr(p2, 'y') else float(p2[1])
    x3 = p3.x() if hasattr(p3, 'x') else float(p3[0])
    y3 = p3.y() if hasattr(p3, 'y') else float(p3[1])

    # Определитель
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return None  # Точки коллинеарны

    # Координаты центра
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d

    try:
        from PyQt6.QtCore import QPointF
        center = QPointF(ux, uy)
    except Exception:
        center = (ux, uy)

    radius = math.hypot(x1 - ux, y1 - uy)
    if radius < 5.0 or radius > 10_000_000.0 or math.isnan(radius) or math.isinf(radius):
        return None

    return center, radius


def fit_circle_3points(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float]
) -> Optional[Tuple[float, float, float]]:
    """
    Аналитически восстанавливает центр (x0, y0) и радиус R окружности по 3 точкам на дуге лимба.
    Для обратной совместимости возвращает кортеж (x0, y0, radius) или None.
    """
    res = circle_from_3_points(p1, p2, p3)
    if res is None:
        return None
    c, r = res
    cx = c.x() if hasattr(c, 'x') else c[0]
    cy = c.y() if hasattr(c, 'y') else c[1]
    return cx, cy, r


def calculate_spherical_correction(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
    scale_km_per_px: float
) -> Dict[str, Any]:
    """
    Вычисляет сферическую коррекцию ракурса (cos θ) для отрезка p1-p2 на диске Луны.

    Вход:
        p1, p2: координаты концов отрезка в пикселях кадра.
        center: центр откалиброванного диска Луны (x0, y0).
        radius: радиус откалиброванного диска в пикселях (R_disk_px).
        scale_km_per_px: масштаб съемки (км/пиксель).

    Выход:
        Словарь с метриками:
        - projected_px: видимая длина в пикселях
        - projected_km: видимая длина в картинной плоскости (км)
        - true_km: истинный топографический размер с учетом сферичности (км) или None (если на/за лимбом)
        - cos_theta: косинус угла наклона поверхности к лучу зрения
        - sin_theta: синус угла наклона
        - tilt_deg: угол наклона поверхности в градусах (0° - субземная точка, 90° - лимб)
        - r_norm: нормализованное удаление от центра Луны (r / R)
        - is_on_disk: True, если центр объекта находится внутри видимого диска
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x0, y0 = float(center[0]), float(center[1])
    r_disk = max(1e-3, float(radius))

    # Длина отрезка в пикселях
    dist_px = math.hypot(x2 - x1, y2 - y1)
    projected_km = dist_px * max(0.0, float(scale_km_per_px))

    # Середина замеряемого объекта
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0

    # Расстояние от центра видимого диска Луны до объекта
    r = math.hypot(mx - x0, my - y0)
    r_norm = r / r_disk

    if r_norm < 1.0:
        sin_theta = r_norm
        cos_theta = math.sqrt(max(0.0, 1.0 - sin_theta * sin_theta))
        tilt_deg = math.degrees(math.asin(sin_theta))
        # Ограничение делителя max(cos_theta, 0.08) для предотвращения ухода в бесконечность у самого лимба (~85.4°)
        cos_clamped = max(cos_theta, 0.08)
        true_km = projected_km / cos_clamped
        is_on_disk = True
    else:
        sin_theta = 1.0
        cos_theta = 0.0
        tilt_deg = 90.0
        true_km = None  # За пределами диска или на самом лимбе
        is_on_disk = False

    return {
        "projected_px": dist_px,
        "projected_km": projected_km,
        "true_km": true_km,
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "tilt_deg": tilt_deg,
        "r_norm": r_norm,
        "is_on_disk": is_on_disk,
    }


def calculate_crater_ellipse(
    p_start: Tuple[float, float],
    p_end: Tuple[float, float],
    cursor_pt: Tuple[float, float],
    scale_km_per_px: float
) -> Dict[str, Any]:
    """
    Рассчитывает параметры эллиптического замера кратера в двухшаговом интерактивном процессе:
    - Шаг 1: p_start и p_end определяют большую ось 2a (истинный диаметр кратера, параллельный лимбу).
    - Шаг 2: cursor_pt задает полумалую ось b (перспективное сжатие вдоль радиального луча к центру диска).

    Выход:
        Словарь с геометрическими и физическими параметрами:
        - center: (x_c, y_c) центр кратера
        - semi_major_a: большая полуось (a) в пикселях
        - semi_minor_b: малая полуось (b) в пикселях
        - angle_deg: угол наклона большой оси (phi) в градусах
        - true_diameter_km: истинный поперечник кратера на поверхности (км) = 2a * scale
        - apparent_diameter_km: сжатый размер вдоль луча зрения (км) = 2b * scale
        - cos_theta: фактор ракурсного сжатия (b / a)
        - tilt_deg: угол наклона поверхности кратера к лучу зрения (градусы)
        - eccentricity: эксцентриситет эллипса sqrt(1 - (b/a)^2)
    """
    x1, y1 = float(p_start[0]), float(p_start[1])
    x2, y2 = float(p_end[0]), float(p_end[1])
    cx_cur, cy_cur = float(cursor_pt[0]), float(cursor_pt[1])

    # Центр кратера
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0

    # Вектор главной оси
    dx = x2 - x1
    dy = y2 - y1
    dist_2a = math.hypot(dx, dy)
    a = max(1e-4, dist_2a / 2.0)

    # Угол поворота главной оси в градусах
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 360.0

    # Единичный перпендикуляр к главной оси (нормаль)
    nx = -dy / dist_2a if dist_2a > 1e-6 else 0.0
    ny = dx / dist_2a if dist_2a > 1e-6 else 1.0

    # Проекция вектора (cursor - center) на нормаль главной оси дает полумалую ось b
    vx = cx_cur - xc
    vy = cy_cur - yc
    proj_b = abs(vx * nx + vy * ny)

    # Полумалая ось не может превышать большую полуось (эллипс не может быть вытянут перпендикулярно)
    b = min(a, max(0.5, proj_b))

    scale = max(0.0, float(scale_km_per_px))
    true_diameter_km = (2.0 * a) * scale
    apparent_diameter_km = (2.0 * b) * scale

    cos_theta = min(1.0, b / a)
    tilt_deg = math.degrees(math.acos(cos_theta))
    eccentricity = math.sqrt(max(0.0, 1.0 - (cos_theta * cos_theta)))

    return {
        "center": (xc, yc),
        "semi_major_a": a,
        "semi_minor_b": b,
        "angle_deg": angle_deg,
        "angle_rad": angle_rad,
        "true_diameter_km": true_diameter_km,
        "apparent_diameter_km": apparent_diameter_km,
        "cos_theta": cos_theta,
        "tilt_deg": tilt_deg,
        "eccentricity": eccentricity,
    }
