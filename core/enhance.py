"""Insert 5 (Local Contrast / Clarity) и Insert 6 (Color Engine & Background Neutralization).

Все функции принимают NumPy-массивы float32 [0..1] (HxW или HxWx3, порядок каналов RGB),
а также uint8 / uint16 — результат возвращается в том же dtype, что и вход.
Входной массив никогда не модифицируется на месте (кэши пайплайна остаются неизменными).

Производительность (кадр 2000x2000): линейные операции выполняются одним проходом cv2.transform,
плавные поля (усиление CLAHE, размытие high-pass, минеральная цветность) считаются на
уменьшенной копии и билинейно растягиваются, проверка гамута — по глобальным min/max с точечным
исправлением только выбивающихся пикселей.
"""
import cv2
import numpy as np

# Коэффициенты яркости Rec.601 (RGB)
_KR, _KG, _KB = 0.299, 0.587, 0.114
_LUMA_ROW = np.array([[_KR, _KG, _KB]], dtype=np.float32)
_LUMA_VEC = np.array([_KR, _KG, _KB], dtype=np.float32)

# Границы рабочих диапазонов параметров
CLAHE_CLIP_RANGE = (1.0, 5.0)
CLAHE_GRID_RANGE = (8, 64)
HIGH_PASS_SIGMA_RANGE = (15.0, 60.0)

# Полоса защиты теней / светов (доля от полной шкалы), где локальный контраст плавно затухает.
# Тени защищаются шире: иначе CLAHE «вытягивает» шумовой фон неба в серый.
_SHADOW_PROTECT = 0.15
_HIGHLIGHT_PROTECT = 0.08

# Максимальный дрейф фона, который может убрать нейтрализация (защита от «срезания» цвета диска)
_BG_MAX_OFFSET = 0.08

# Кадры крупнее этого размера обрабатываются полями на половинном разрешении
_FIELD_DOWNSCALE_PIXELS = 1_500_000
# Целевой размер (пикселей) низкоразрешающей копии для расчёта минеральной цветности
_MINERAL_FIELD_PIXELS = 500_000


# =============================================================================
# Вспомогательные функции
# =============================================================================

def _to_float32(image: np.ndarray):
    """Возвращает (float32 [0..1], исходный dtype). Для float32 копия не создаётся."""
    arr = np.asarray(image)
    dt = arr.dtype
    if dt == np.uint8:
        f = arr.astype(np.float32)
        f *= np.float32(1.0 / 255.0)
    elif dt == np.uint16:
        f = arr.astype(np.float32)
        f *= np.float32(1.0 / 65535.0)
    else:
        f = np.ascontiguousarray(arr, dtype=np.float32)
    return f, dt


def _from_float32(f: np.ndarray, dtype) -> np.ndarray:
    """Обратное преобразование float32 [0..1] -> исходный dtype (с насыщением и округлением)."""
    if dtype == np.uint8:
        np.clip(f, 0.0, 1.0, out=f)
        return cv2.convertScaleAbs(f, alpha=255.0)
    if dtype == np.uint16:
        np.clip(f, 0.0, 1.0, out=f)
        return (f * 65535.0 + 0.5).astype(np.uint16)
    if dtype == np.float32:
        return f
    return f.astype(dtype)


def _axpb(x: np.ndarray, a: float, b: float = 0.0, dst=None) -> np.ndarray:
    """dst = a * x + b (одна многопоточная операция OpenCV)."""
    return cv2.addWeighted(x, float(a), x, 0.0, float(b), dst=dst)


def _clip01_(p: np.ndarray) -> np.ndarray:
    """Обрезка массива (любого числа каналов) в [0, 1] на месте."""
    cv2.threshold(p, 1.0, 1.0, cv2.THRESH_TRUNC, dst=p)
    cv2.threshold(p, 0.0, 0.0, cv2.THRESH_TOZERO, dst=p)
    return p


def _clip01_if_needed_(p: np.ndarray) -> np.ndarray:
    """Обрезка в [0, 1] только если есть выход за границы (две дешёвые редукции)."""
    if float(p.max()) > 1.0 or float(p.min()) < 0.0:
        _clip01_(p)
    return p


def _gray3(plane: np.ndarray) -> np.ndarray:
    """Одноканальная карта -> 3 одинаковых канала (для поканального умножения/сложения)."""
    return cv2.cvtColor(plane, cv2.COLOR_GRAY2RGB)


def _luma_plane(f3: np.ndarray) -> np.ndarray:
    """Карта яркости Rec.601 из HxWx3 за один проход."""
    h, w = f3.shape[:2]
    return cv2.transform(f3, _LUMA_ROW).reshape(h, w)


def _protect_lut(size: int = 65536) -> np.ndarray:
    """LUT весов защиты: 0 в чёрном/белом, плавно растёт до 1 в полутонах."""
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    lo = np.minimum(x / np.float32(_SHADOW_PROTECT), 1.0)
    hi = np.minimum((1.0 - x) / np.float32(_HIGHLIGHT_PROTECT), 1.0)
    w = np.clip(lo * hi, 0.0, 1.0)
    return (w * w * (3.0 - 2.0 * w)).astype(np.float32)  # smoothstep


_PROTECT_LUT = _protect_lut()


def _to_u16(y: np.ndarray) -> np.ndarray:
    """float32 [0..1] -> uint16 (с округлением и насыщением)."""
    return cv2.addWeighted(y, 65535.0, y, 0.0, 0.0, dtype=cv2.CV_16U)


def _protect_weight(y: np.ndarray) -> np.ndarray:
    """Вес защиты теней/светов для карты яркости (float32 [0..1])."""
    return _PROTECT_LUT[_to_u16(y)]


def _has_color(f: np.ndarray) -> bool:
    return f.ndim == 3 and f.shape[2] >= 3


def _rgb3(f: np.ndarray) -> np.ndarray:
    """Первые три канала как непрерывный HxWx3 float32."""
    return f if f.shape[2] == 3 else np.ascontiguousarray(f[:, :, :3])


def _resize(img: np.ndarray, w: int, h: int, area: bool = False) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if area else cv2.INTER_LINEAR)


# =============================================================================
# INSERT 5: Local Contrast / Clarity
# =============================================================================

def _lc_clahe(f: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """CLAHE по яркости (luma) с сохранением цвета и защитой светов/теней."""
    clip = float(np.clip(amount, *CLAHE_CLIP_RANGE))
    grid = int(round(float(np.clip(radius, *CLAHE_GRID_RANGE))))
    h, w = f.shape[:2]

    color = _has_color(f)
    if color:
        f3 = _rgb3(f)
        y = _luma_plane(f3)
        _clip01_(y)
    else:
        y = np.clip(f if f.ndim == 2 else f[:, :, 0], 0.0, 1.0)  # новая копия: вход не трогаем

    # Поле усиления считаем на половинном разрешении для крупных кадров
    ds = 2 if h * w > _FIELD_DOWNSCALE_PIXELS else 1
    ys = _resize(y, w // ds, h // ds, area=True) if ds > 1 else y

    # CLAHE на 8-битной яркости: именно при 256 бинах clipLimit имеет стандартный смысл
    # (при 16 битах и мелких тайлах предел отсечения вырождается в 1 и «amount» перестаёт работать).
    # Результат используется как карта усиления в float, поэтому квантование на выходе не видно.
    ys8 = cv2.addWeighted(ys, 255.0, ys, 0.0, 0.0, dtype=cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    yc8 = clahe.apply(ys8)

    # Изменение яркости от CLAHE (yc - y), с защитой теней и светов: эффект затухает у 0 и 1
    d = cv2.addWeighted(yc8, 1.0 / 255.0, ys8, -1.0 / 255.0, 0.0, dtype=cv2.CV_32F)
    cv2.multiply(d, _protect_weight(ys), dst=d)
    yns = cv2.add(ys, d)

    gain = cv2.divide(yns, np.maximum(ys, np.float32(1e-4)))
    np.clip(gain, 0.25, 4.0, out=gain)
    if ds > 1:
        gain = _resize(gain, w, h)

    if not color:
        return _clip01_(cv2.multiply(y, gain))

    # Мультипликативный перенос яркости на RGB: оттенок и насыщенность сохраняются
    out = cv2.multiply(f3, _gray3(gain))
    _limit_gamut_(out)   # усиление могло вывести канал за 1.0 — снижаем цветность, не обрезаем оттенок
    return out


def _blur_field(y: np.ndarray, sigma: float):
    """Гауссово размытие (большой sigma) и вес защиты, посчитанные на уменьшенной копии."""
    h, w = y.shape[:2]
    ds = max(2, int(sigma // 6))
    sw, sh = max(2, w // ds), max(2, h // ds)
    small = _resize(y, sw, sh, area=True)
    blur_s = cv2.GaussianBlur(small, (0, 0), sigma * sw / float(w), borderType=cv2.BORDER_REFLECT)
    wt_s = _protect_weight(np.clip(blur_s, 0.0, 1.0))   # вес по низким частотам: без шума
    return blur_s, wt_s, (sw, sh)


def _lc_high_pass(f: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """Image + amount * (Image - Blurred) по яркости, с защитой светов и теней."""
    sigma = float(np.clip(radius, *HIGH_PASS_SIGMA_RANGE))
    amount = float(max(0.0, amount))
    h, w = f.shape[:2]

    color = _has_color(f)
    if color:
        f3 = _rgb3(f)
        y = _luma_plane(f3)
    else:
        y = f if f.ndim == 2 else f[:, :, 0]

    blur_s, wt_s, _ = _blur_field(y, sigma)
    _axpb(wt_s, amount, 0.0, dst=wt_s)                  # amount вшит в маску защиты

    delta = cv2.subtract(y, _resize(blur_s, w, h))
    cv2.multiply(delta, _resize(wt_s, w, h), dst=delta)

    if not color:
        return _clip01_(cv2.add(y, delta))

    out = cv2.add(f3, _gray3(delta))
    return _clip01_if_needed_(out)


def apply_local_contrast(image: np.ndarray, amount: float, radius: float,
                         method: str = 'clahe') -> np.ndarray:
    """Insert 5: локальный контраст / Clarity.

    method='clahe':
        CLAHE по каналу яркости. amount -> clipLimit (1.0–5.0), radius -> размер сетки
        тайлов N×N (8–64; больше N = мельче «локальность»). Цвет сохраняется переносом
        отношения яркостей на RGB; у чёрного и белого эффект плавно затухает.
    method='high_pass':
        Image + amount * (Image - Blurred), sigma = radius (15–60 px). Размытие считается
        на уменьшенной копии; разница подмешивается к яркости с защитой светов/теней.

    Вход/выход: float32 [0..1] (или uint8/uint16 — dtype сохраняется), HxW или HxWx3.
    """
    arr = np.asarray(image)
    method = str(method).lower().replace('-', '_')

    if method in ('high_pass', 'highpass', 'hp'):
        if float(amount) <= 1e-4:
            return arr.copy()
        f, dt = _to_float32(arr)
        return _from_float32(_lc_high_pass(f, amount, radius), dt)

    if method != 'clahe':
        raise ValueError(f"Неизвестный метод local contrast: {method!r} (ожидается 'clahe' или 'high_pass')")

    if float(amount) <= CLAHE_CLIP_RANGE[0] + 1e-4:
        return arr.copy()  # clipLimit 1.0 — нейтраль
    f, dt = _to_float32(arr)
    return _from_float32(_lc_clahe(f, amount, radius), dt)


# =============================================================================
# INSERT 6: Color Engine & Background Neutralization
# =============================================================================

def _bg_region_mean(f3: np.ndarray, bg_coords):
    """Средние (R, G, B) по прямоугольнику bg_coords = (x, y, w, h) или (x, y) -> бокс 32×32."""
    h, w = f3.shape[:2]
    c = tuple(bg_coords)
    if len(c) == 2:
        bw = bh = 32
        x0, y0 = int(round(c[0] - bw / 2)), int(round(c[1] - bh / 2))
    elif len(c) == 4:
        x0, y0, bw, bh = (int(round(v)) for v in c)
    else:
        return None
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x0 + max(1, bw)), min(h, y0 + max(1, bh))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return f3[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


def _bg_auto_sample(f3: np.ndarray):
    """Авто-семпл фона: средние (R, G, B) по самым тёмным ~10% пикселей (небо вне диска)."""
    h, w = f3.shape[:2]
    step = max(1, int(np.sqrt(h * w / 250_000.0)))
    s = f3[::step, ::step].reshape(-1, 3)
    lum = s @ _LUMA_VEC
    thr = min(float(np.percentile(lum, 10.0)), 0.25)
    mask = lum <= thr
    if int(mask.sum()) < 16:
        return None
    return s[mask].mean(axis=0)


def _estimate_bg_offsets(f3: np.ndarray, bg_coords) -> np.ndarray:
    """Хроматический дрейф фона (R_offset, G_offset, B_offset).

    off_c = bg_c - min(bg): канал с минимальным фоном не трогается, остальные опускаются
    до него, общий уровень фона (тёмно-серый) и тени терминатора остаются нетронутыми.
    """
    bg = None
    if bg_coords is not None:
        bg = _bg_region_mean(f3, bg_coords)
    if bg is None:
        bg = _bg_auto_sample(f3)
    if bg is None:
        return np.zeros(3, dtype=np.float64)
    bg = np.asarray(bg, dtype=np.float64)
    return np.clip(bg - bg.min(), 0.0, _BG_MAX_OFFSET)


def _limit_gamut_(img: np.ndarray) -> None:
    """Ограничивает цвет в гамуте [0, 1] на месте.

    - Отрицательные значения (шум в тёмном фоне) просто обнуляются: оттенок там не важен.
    - Пересветы (канал > 1) устраняются уменьшением цветности пикселя с сохранением его
      яркости и оттенка. Глобальные min/max дёшевы; исправляются только выбивающиеся пиксели.
    """
    if float(img.min()) < 0.0:
        cv2.threshold(img, 0.0, 0.0, cv2.THRESH_TOZERO, dst=img)
    if float(img.max()) <= 1.0:
        return

    h, w = img.shape[:2]
    flat = img.reshape(-1, 3)
    idx = np.flatnonzero(cv2.reduce(flat, 1, cv2.REDUCE_MAX).ravel() > 1.0)
    if idx.size == 0:
        return

    # Работаем по строкам (3, K): все операции над непрерывными векторами
    rgb = np.ascontiguousarray(flat[idx].T)
    r, g, b = rgb
    y = _KR * r + _KG * g + _KB * b
    mx = np.maximum(np.maximum(r, g), b)
    t = np.clip((1.0 - y) / np.maximum(mx - y, np.float32(1e-6)), 0.0, 1.0)
    rgb = y + (rgb - y) * t           # y + (p - y) * t
    np.clip(rgb, 0.0, 1.0, out=rgb)
    flat[idx] = rgb.T


def _mineral_boost_(img: np.ndarray, boost: float) -> None:
    """Избирательное нелинейное насыщение синих (титан) и оранжево-красных (оксиды Fe) тонов.

    Ось «холодный–тёплый» t = |B - R|: чем больше t относительно яркости, тем сильнее усиление
    цветности R и B (насыщающаяся кривая s = x / (x + 0.15)); малоцветные и тёмные (шумовые)
    пиксели почти не затрагиваются. Паразитный зелёный (G выше среднего R и B) вычитается.
    Поправка вычисляется на низкоразрешающей копии (цветность — низкочастотное поле), затем
    растягивается и прибавляется к кадру; яркость поправки обнуляется (luma сохраняется).
    """
    boost = float(np.clip(boost, 0.0, 2.0))
    K = 2.5
    sup = 0.8 * min(boost, 1.0)

    h, w = img.shape[:2]
    ds = max(1, int(np.ceil(np.sqrt(h * w / float(_MINERAL_FIELD_PIXELS)))))
    small = _resize(img, max(2, w // ds), max(2, h // ds), area=True) if ds > 1 else img
    r, g, b = cv2.split(small)

    y = cv2.addWeighted(r, _KR, g, _KG, 0.0)
    cv2.addWeighted(y, 1.0, b, _KB, 0.0, dst=y)

    tn = cv2.divide(cv2.absdiff(b, r), _axpb(y, 1.0, 0.05))
    s = cv2.divide(tn, _axpb(tn, 1.0, 0.15))               # 0..1, насыщающийся отклик на цветность
    m = cv2.divide(y, _axpb(y, 1.0, 0.04))                  # затухание в тёмном фоне (шум)
    g1 = cv2.multiply(s, m)
    _axpb(g1, K * boost, 0.0, dst=g1)                       # gain - 1

    dr = cv2.multiply(cv2.subtract(r, y), g1)
    db = cv2.multiply(cv2.subtract(b, y), g1)

    # Подавление зелёного: gs = max(G - (R' + B') / 2, 0), R' = R + dr, B' = B + db
    rb = cv2.addWeighted(cv2.add(r, dr), 0.5, cv2.add(b, db), 0.5, 0.0)
    gs = cv2.subtract(g, rb)
    cv2.threshold(gs, 0.0, 0.0, cv2.THRESH_TOZERO, dst=gs)
    cv2.multiply(gs, m, dst=gs)
    dg = _axpb(gs, -sup, 0.0)

    # Яркость поправки = 0: вычитаем luma(dr, dg, db) из всех каналов
    dl = cv2.addWeighted(dr, _KR, dg, _KG, 0.0)
    cv2.addWeighted(dl, 1.0, db, _KB, 0.0, dst=dl)
    for p in (dr, dg, db):
        cv2.subtract(p, dl, dst=p)

    delta = cv2.merge([dr, dg, db])
    if ds > 1:
        delta = _resize(delta, w, h)
    cv2.add(img, delta, dst=img)


def apply_color_engine(image: np.ndarray, saturation: float = 1.0, mineral_boost: float = 0.0,
                       bg_neutralize: bool = False, bg_coords=None) -> np.ndarray:
    """Insert 6: насыщенность, Mineral Moon Boost и нейтрализация фона.

    - Ч/б изображение (HxW или HxWx1): возвращается без изменений (без ошибок).
    - saturation: множитель цветности (1.0 — нейтраль); яркость L сохраняется точно,
      пересветы подавляются уменьшением цветности пикселя, а не обрезкой каналов.
    - mineral_boost (0..~1): титан (синий) / оксиды железа (оранжево-красный) усиливаются
      нелинейно, паразитный зелёный подавляется.
    - bg_neutralize: фон приводится к нейтральному серому. bg_coords = (x, y, w, h) либо
      (x, y) (бокс 32×32); если None — авто-семпл самых тёмных пикселей.

    Вход/выход: float32 [0..1] (или uint8/uint16 — dtype сохраняется), порядок каналов RGB.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr

    saturation = float(saturation)
    mineral_boost = float(mineral_boost)
    do_sat = abs(saturation - 1.0) > 1e-3
    do_min = mineral_boost > 1e-3
    if not (do_sat or do_min or bg_neutralize):
        return arr

    f, dt = _to_float32(arr)
    f3 = _rgb3(f)

    # Нейтрализация фона и насыщенность — линейные: сводим в одну матрицу 3x4 (один проход)
    A = np.eye(3, dtype=np.float64)
    c = np.zeros(3, dtype=np.float64)
    if bg_neutralize:
        off = _estimate_bg_offsets(f3, bg_coords)
        A = np.diag(1.0 / (1.0 - off))                  # (p - off) / (1 - off): белая точка сохранена
        c = -off / (1.0 - off)
    if do_sat:
        s = max(0.0, saturation)
        S = s * np.eye(3) + (1.0 - s) * np.outer(np.ones(3), _LUMA_VEC)   # Y + (p - Y) * s
        A, c = S @ A, S @ c

    if bg_neutralize or do_sat:
        out = cv2.transform(f3, np.hstack([A, c[:, None]]).astype(np.float32))
    else:
        out = f3.copy()

    if do_min:
        _mineral_boost_(out, mineral_boost)

    _limit_gamut_(out)

    if f.shape[2] > 3:  # дополнительные каналы (например, alpha) возвращаем нетронутыми
        out = np.concatenate([out, f[:, :, 3:]], axis=2)
    return _from_float32(out, dt)
