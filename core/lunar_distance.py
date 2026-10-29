"""
Аналитический модуль расчёта геоцентрического расстояния до Луны.
Основан на усечённой тригонометрической теории Жана Меёса (Jean Meeus, Astronomical Algorithms)
и теории ELP-2000. Не требует внешних астрономических зависимостей (только math и datetime).
"""

import math
import datetime
from typing import Union, Tuple, Dict

# Среднее расстояние до Луны (км)
DEFAULT_LUNAR_DISTANCE_KM = 384400.0
MEAN_LUNAR_DISTANCE_KM = 385000.56

# Основные периодические члены Меёса (Глава 47, Таблица 47.A) для расстояния Delta (в км).
# Кортежи формата: (D, M, M', F, coefficient_km)
_LUNAR_DISTANCE_TERMS: Tuple[Tuple[int, int, int, int, float], ...] = (
    # Главные гармоники (эллиптические возмущения, эвекция, вариация и др.)
    (0, 0, 1, 0, -20905.355),   # cos(M') - главное эллиптическое уравнение
    (2, 0, -1, 0, -3699.111),   # cos(2D - M') - главная эвекция
    (2, 0, 0, 0, -2955.968),    # cos(2D) - главная вариация
    (0, 0, 2, 0, -569.925),     # cos(2M') - вторичное эллиптическое
    (0, 1, 0, 0, 48.888),       # cos(M)
    (0, 0, 0, 2, -3.149),       # cos(2F)
    (2, 0, -2, 0, 246.158),     # cos(2D - 2M')
    (2, -1, -1, 0, -152.138),   # cos(2D - M - M')
    (2, 0, 1, 0, -170.733),     # cos(2D + M')
    (2, -1, 0, 0, -204.586),    # cos(2D - M)
    (0, 1, -1, 0, -129.620),    # cos(M - M')
    (1, 0, 0, 0, 108.743),      # cos(D)
    (0, 1, 1, 0, 104.755),      # cos(M + M')
    (2, 0, 0, -2, 10.321),      # cos(2D - 2F)
    (0, 0, 1, -2, 79.661),      # cos(M' - 2F)
    (4, 0, -1, 0, -34.782),     # cos(4D - M')
    (0, 0, 3, 0, -23.210),      # cos(3M')
    (4, 0, -2, 0, -21.636),     # cos(4D - 2M')
    (2, 1, -1, 0, 24.208),      # cos(2D + M - M')
    (2, 1, 0, 0, 30.824),       # cos(2D + M)
    (1, 0, -1, 0, -8.379),      # cos(D - M')
    (1, 1, 0, 0, -16.675),      # cos(D + M)
    (2, -1, 1, 0, -12.831),     # cos(2D - M + M')
    (2, 0, 2, 0, -10.445),      # cos(2D + 2M')
    (4, 0, 0, 0, -11.650),      # cos(4D)
    (2, 0, -3, 0, 14.403),      # cos(2D - 3M')
    (0, 1, -2, 0, -7.003),      # cos(M - 2M')
    (2, -1, -2, 0, 10.056),     # cos(2D - M - 2M')
    (1, 0, 1, 0, 6.322),        # cos(D + M')
    (2, -2, 0, 0, -9.884),      # cos(2D - 2M)
    (0, 1, 2, 0, 5.751),        # cos(M + 2M')
    (2, -2, -1, 0, -4.950),     # cos(2D - 2M - M')
    (2, 0, 1, -2, 4.130),       # cos(2D + M' - 2F)
)


def date_to_julian_day(date_or_dt: Union[datetime.date, datetime.datetime]) -> float:
    """
    Преобразует дату или дату-время в юлианский день (JD).
    Если передан datetime с временной зоной, он нормализуется в UTC.
    """
    if isinstance(date_or_dt, datetime.datetime):
        if date_or_dt.tzinfo is not None:
            utc_dt = date_or_dt.astimezone(datetime.timezone.utc)
        else:
            utc_dt = date_or_dt
        day_fraction = (
            utc_dt.day
            + (utc_dt.hour + (utc_dt.minute + (utc_dt.second + utc_dt.microsecond / 1e6) / 60.0) / 60.0) / 24.0
        )
        y, m = utc_dt.year, utc_dt.month
    elif isinstance(date_or_dt, datetime.date):
        day_fraction = float(date_or_dt.day)
        y, m = date_or_dt.year, date_or_dt.month
    else:
        raise TypeError(f"Ожидался datetime.date или datetime.datetime, получено: {type(date_or_dt)}")

    if m <= 2:
        y -= 1
        m += 12

    a = int(y / 100)
    b = 2 - a + int(a / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day_fraction + b - 1524.5
    return jd


def get_lunar_orbital_arguments(t_centuries: float) -> Dict[str, float]:
    """
    Вычисляет фундаментальные аргументы теории движения Луны Меёса для заданного
    количества юлианских столетий от эпохи J2000.0 (T).
    Все углы возвращаются в градусах в диапазоне [0, 360).
    """
    t = t_centuries
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t

    # Средняя долгота Луны L'
    lp = 218.3164477 + 481267.88123421 * t - 0.0015786 * t2 + t3 / 538841.0 - t4 / 65194000.0
    # Средняя элонгация Луны D
    d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t2 + t3 / 545868.0 - t4 / 113065000.0
    # Средняя аномалия Солнца M
    m_sun = 357.5291092 + 35999.0502909 * t - 0.0001536 * t2 + t3 / 24490000.0
    # Средняя аномалия Луны M'
    m_moon = 134.9633964 + 477198.8675055 * t + 0.0087414 * t2 + t3 / 69699.0 - t4 / 14712000.0
    # Аргумент широты Луны F
    f = 93.2720950 + 483202.0175233 * t - 0.0036539 * t2 - t3 / 3526000.0 + t4 / 863310000.0

    return {
        "L_prime": lp % 360.0,
        "D": d % 360.0,
        "M": m_sun % 360.0,
        "M_prime": m_moon % 360.0,
        "F": f % 360.0,
    }


def calculate_lunar_distance(
    date_or_dt: Union[datetime.date, datetime.datetime],
    round_digits: int = 1
) -> float:
    """
    Рассчитывает актуальное геоцентрическое расстояние до Луны (Delta) в километрах
    для заданной даты съемки по усечённой тригонометрической теории Меёса.

    Вход:
        date_or_dt: datetime.date или datetime.datetime (UTC)
        round_digits: количество знаков после запятой (по умолчанию 1)

    Выход:
        Расстояние до Луны в километрах (например, 363300 .. 405500 км).
        Точность не хуже ±50-100 км от полных эфемерид DE430/DE440.
    """
    jd = date_to_julian_day(date_or_dt)
    t = (jd - 2451545.0) / 36525.0

    args = get_lunar_orbital_arguments(t)
    d_rad = math.radians(args["D"])
    m_sun_rad = math.radians(args["M"])
    m_moon_rad = math.radians(args["M_prime"])
    f_rad = math.radians(args["F"])

    # Коэффициент эксцентриситета орбиты Земли E
    e_factor = 1.0 - 0.002516 * t - 0.0000074 * (t ** 2)

    delta_km = MEAN_LUNAR_DISTANCE_KM

    for d_mul, m_mul, mp_mul, f_mul, coeff in _LUNAR_DISTANCE_TERMS:
        angle_rad = d_mul * d_rad + m_mul * m_sun_rad + mp_mul * m_moon_rad + f_mul * f_rad
        term_coeff = coeff
        if abs(m_mul) == 1:
            term_coeff *= e_factor
        elif abs(m_mul) == 2:
            term_coeff *= (e_factor ** 2)

        delta_km += term_coeff * math.cos(angle_rad)

    if round_digits <= 0:
        return float(round(delta_km))
    return float(round(delta_km, round_digits))
