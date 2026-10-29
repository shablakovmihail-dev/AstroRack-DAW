import unittest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry_calc import (
    fit_circle_3points,
    calculate_spherical_correction,
    calculate_crater_ellipse
)


class TestGeometryCalc(unittest.TestCase):
    def test_fit_circle_exact(self):
        # Окружность с центром (200, 300) и радиусом R = 150
        cx, cy, r_true = 200.0, 300.0, 150.0
        p1 = (cx + r_true * math.cos(0.0), cy + r_true * math.sin(0.0))
        p2 = (cx + r_true * math.cos(2.0), cy + r_true * math.sin(2.0))
        p3 = (cx + r_true * math.cos(4.0), cy + r_true * math.sin(4.0))

        res = fit_circle_3points(p1, p2, p3)
        self.assertIsNotNone(res)
        x0, y0, r_calc = res
        self.assertAlmostEqual(x0, cx, places=3)
        self.assertAlmostEqual(y0, cy, places=3)
        self.assertAlmostEqual(r_calc, r_true, places=3)

    def test_fit_circle_right_triangle(self):
        # Точки на диаметре (-100, 0) и (100, 0) и верхняя точка (0, 100) -> центр (0, 0), R = 100
        res = fit_circle_3points((-100.0, 0.0), (0.0, 100.0), (100.0, 0.0))
        self.assertIsNotNone(res)
        x0, y0, r = res
        self.assertAlmostEqual(x0, 0.0, places=4)
        self.assertAlmostEqual(y0, 0.0, places=4)
        self.assertAlmostEqual(r, 100.0, places=4)

    def test_fit_circle_collinear_fails_safely(self):
        # 3 точки на одной линии не образуют окружность
        self.assertIsNone(fit_circle_3points((0.0, 0.0), (10.0, 10.0), (20.0, 20.0)))
        self.assertIsNone(fit_circle_3points((50.0, 100.0), (50.0, 200.0), (50.0, 300.0)))

    def test_fit_circle_coincident_points(self):
        # Две совпадающие точки
        self.assertIsNone(fit_circle_3points((50.0, 50.0), (50.0, 50.0), (100.0, 100.0)))

    def test_spherical_correction_subearth_center(self):
        # Объект прямо в центре видимого диска Луны (θ = 0°, cos θ = 1.0)
        center = (500.0, 500.0)
        radius = 400.0
        p1 = (490.0, 500.0)
        p2 = (510.0, 500.0)
        scale_km_per_px = 1.5  # км/px

        res = calculate_spherical_correction(p1, p2, center, radius, scale_km_per_px)
        self.assertTrue(res["is_on_disk"])
        self.assertAlmostEqual(res["projected_px"], 20.0, places=4)
        self.assertAlmostEqual(res["projected_km"], 30.0, places=4)
        self.assertAlmostEqual(res["cos_theta"], 1.0, places=4)
        self.assertAlmostEqual(res["tilt_deg"], 0.0, delta=0.1)
        self.assertAlmostEqual(res["true_km"], 30.0, places=3)

    def test_spherical_correction_sixty_degrees(self):
        # Объект на удалении r = R * sin(60°) = R * 0.866025
        center = (0.0, 0.0)
        radius = 1000.0
        r_dist = 1000.0 * math.sin(math.radians(60.0))  # 866.025 px
        p1 = (r_dist, -25.0)
        p2 = (r_dist, 25.0)  # L = 50 px, середина строго на r_dist
        scale_km_per_px = 1.0

        res = calculate_spherical_correction(p1, p2, center, radius, scale_km_per_px)
        self.assertTrue(res["is_on_disk"])
        self.assertAlmostEqual(res["projected_km"], 50.0, places=3)
        self.assertAlmostEqual(res["cos_theta"], 0.5, delta=0.01)
        self.assertAlmostEqual(res["tilt_deg"], 60.0, delta=0.5)
        # При cos θ = 0.5 истинный размер должен быть в 2 раза больше проекции: 50 / 0.5 = 100 км!
        self.assertAlmostEqual(res["true_km"], 100.0, delta=1.0)

    def test_spherical_correction_outside_limb(self):
        # Объект за пределами диска Луны
        center = (0.0, 0.0)
        radius = 500.0
        p1 = (550.0, 0.0)
        p2 = (570.0, 0.0)
        res = calculate_spherical_correction(p1, p2, center, radius, 1.0)
        self.assertFalse(res["is_on_disk"])
        self.assertIsNone(res["true_km"])
        self.assertEqual(res["tilt_deg"], 90.0)

    def test_crater_ellipse_simple(self):
        # Большая ось от (0, 0) до (100, 0) -> a = 50. Курсор в (50, 25) -> b = 25
        scale_km = 2.0
        p_start = (0.0, 0.0)
        p_end = (100.0, 0.0)
        cursor_pt = (50.0, 25.0)

        res = calculate_crater_ellipse(p_start, p_end, cursor_pt, scale_km)
        self.assertEqual(res["center"], (50.0, 0.0))
        self.assertAlmostEqual(res["semi_major_a"], 50.0, places=4)
        self.assertAlmostEqual(res["semi_minor_b"], 25.0, places=4)
        self.assertAlmostEqual(res["true_diameter_km"], 200.0, places=3)  # 2a * 2.0 = 200.0 км
        self.assertAlmostEqual(res["apparent_diameter_km"], 100.0, places=3)  # 2b * 2.0 = 100.0 км
        self.assertAlmostEqual(res["cos_theta"], 0.5, places=4)
        self.assertAlmostEqual(res["tilt_deg"], 60.0, delta=0.1)
        # e = sqrt(1 - 0.5^2) = sqrt(0.75) ≈ 0.866
        self.assertAlmostEqual(res["eccentricity"], math.sqrt(0.75), places=3)

    def test_crater_ellipse_rotated(self):
        # Эллипс повернут под углом 90 градусов (вертикальная главная ось)
        p_start = (100.0, 0.0)
        p_end = (100.0, 80.0)  # a = 40, угол 90°
        cursor_pt = (120.0, 40.0)  # b = 20
        res = calculate_crater_ellipse(p_start, p_end, cursor_pt, 1.0)

        self.assertAlmostEqual(res["semi_major_a"], 40.0, places=4)
        self.assertAlmostEqual(res["semi_minor_b"], 20.0, places=4)
        self.assertAlmostEqual(res["angle_deg"], 90.0, delta=0.1)
        self.assertAlmostEqual(res["cos_theta"], 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
