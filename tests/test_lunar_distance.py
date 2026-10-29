import unittest
import datetime
from core.lunar_distance import (
    date_to_julian_day,
    get_lunar_orbital_arguments,
    calculate_lunar_distance,
    DEFAULT_LUNAR_DISTANCE_KM,
    MEAN_LUNAR_DISTANCE_KM
)


class TestLunarDistance(unittest.TestCase):
    def test_julian_day_j2000(self):
        # Эпоха J2000.0: 1 января 2000 года, 12:00:00 TT/UTC
        dt_j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        jd = date_to_julian_day(dt_j2000)
        self.assertAlmostEqual(jd, 2451545.0, places=5)

    def test_julian_day_date_only(self):
        d = datetime.date(2026, 10, 29)
        jd = date_to_julian_day(d)
        # 2026-10-29 00:00 UTC -> 2461342.5
        self.assertAlmostEqual(jd, 2461342.5, places=3)

    def test_julian_day_timezone_conversion(self):
        # 12:00 UTC+5 эквивалентно 07:00 UTC
        tz_plus5 = datetime.timezone(datetime.timedelta(hours=5))
        dt1 = datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=tz_plus5)
        dt_utc = datetime.datetime(2024, 6, 1, 7, 0, 0, tzinfo=datetime.timezone.utc)
        self.assertAlmostEqual(date_to_julian_day(dt1), date_to_julian_day(dt_utc), places=6)

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            date_to_julian_day("2024-05-01")

    def test_orbital_arguments_range(self):
        args = get_lunar_orbital_arguments(0.24)  # ~2024 год
        for key in ("L_prime", "D", "M", "M_prime", "F"):
            self.assertIn(key, args)
            self.assertGreaterEqual(args[key], 0.0)
            self.assertLess(args[key], 360.0)

    def test_meeus_example_1992(self):
        # Пример из книги Жана Меёса "Astronomical Algorithms" (2-е изд., стр. 342)
        # 12 апреля 1992 г., 0h TD -> ~368410 км
        dt = datetime.datetime(1992, 4, 12, 0, 0, 0, tzinfo=datetime.timezone.utc)
        dist = calculate_lunar_distance(dt, round_digits=1)
        self.assertAlmostEqual(dist, 368418.4, delta=20.0)

    def test_supermoon_perigee(self):
        # Суперлуние (перигей) 30 августа 2023 года (~357 300 км)
        dt = datetime.date(2023, 8, 30)
        dist = calculate_lunar_distance(dt)
        # Расстояние в перигее должно быть в диапазоне 356 000 .. 365 000 км
        self.assertGreaterEqual(dist, 356000.0)
        self.assertLessEqual(dist, 365000.0)

    def test_micromoon_apogee(self):
        # Микролуние (апогей) 4 февраля 2023 года (~406 000 км)
        dt = datetime.date(2023, 2, 4)
        dist = calculate_lunar_distance(dt)
        # Расстояние в апогее должно быть в диапазоне 404 000 .. 407 500 км
        self.assertGreaterEqual(dist, 404000.0)
        self.assertLessEqual(dist, 407500.0)

    def test_rounding_parameter(self):
        dt = datetime.date(2024, 1, 15)
        dist_1 = calculate_lunar_distance(dt, round_digits=1)
        dist_0 = calculate_lunar_distance(dt, round_digits=0)
        self.assertEqual(dist_0, round(dist_1))


if __name__ == "__main__":
    unittest.main()
