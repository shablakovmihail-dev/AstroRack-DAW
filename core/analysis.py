#"""Аналитические инструменты: Star PSF & FWHM Analyzer и Pixel & Angular Distance Calculator."""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~ 1 / 2.354820045
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))          # ~ 2.354820045
ARCSEC_RAD_FACTOR = 206.265


class AnalysisResult(dict):
    """Словарь результатов анализа с поддержкой доступа через атрибуты (res.fwhm и res['fwhm'])."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"'AnalysisResult' object has no attribute '{name}'") from exc

    def __setattr__(self, name: str, value):
        self[name] = value


def _to_luminance_float32(image_array: np.ndarray) -> np.ndarray:
    """Преобразует входной массив (2D или 3D RGB) в 2D float32 карту яркости."""
    arr = np.asarray(image_array, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    return arr


def _gaussian_1d(r: np.ndarray, amp: float, sigma: float) -> np.ndarray:
    """Радиальная 1D функция Гаусса (после вычитания фона)."""
    return amp * np.exp(-0.5 * (r / np.maximum(sigma, 1e-4)) ** 2)


def _moffat_1d(r: np.ndarray, amp: float, alpha: float, beta: float) -> np.ndarray:
    """Радиальная 1D функция Моффата (после вычитания фона)."""
    alpha_safe = np.maximum(alpha, 1e-4)
    beta_safe = np.maximum(beta, 0.5)
    return amp * np.power(1.0 + (r / alpha_safe) ** 2, -beta_safe)


def calculate_fwhm(
    image_array: np.ndarray,
    center_x: float,
    center_y: float,
    box_size: int = 31
) -> AnalysisResult:
    """
    Анализатор профиля звезды (Star PSF & FWHM Analyzer).

    1. Вырезает ROI размером `box_size x box_size` вокруг `(center_x, center_y)`.
    2. Оценивает уровень локального фона (Background) и шума по внешнему кольцу ROI.
    3. Находит точный субпиксельный центроид звезды (взвешенный по потоку центр масс).
    4. Аппроксимирует радиальный профиль функцией Гаусса и функцией Моффата (Moffat).
    5. Возвращает FWHM (px), Sigma (px), Peak, Background, 1D-срез (Slice) для графика
       и рекомендуемый параметр `recommended_rl_sigma` для деконволюции Ричардсона-Люси.
    """
    if image_array is None or image_array.size == 0:
        raise ValueError("image_array пуст или не задан.")

    lum = _to_luminance_float32(image_array)
    h, w = lum.shape[:2]

    box_size = max(7, int(box_size))
    if box_size % 2 == 0:
        box_size += 1
    half = box_size // 2

    cx_init = float(np.clip(center_x, 0.0, max(0.0, w - 1.0)))
    cy_init = float(np.clip(center_y, 0.0, max(0.0, h - 1.0)))

    ix = int(round(cx_init))
    iy = int(round(cy_init))

    x0 = max(0, ix - half)
    x1 = min(w, ix + half + 1)
    y0 = max(0, iy - half)
    y1 = min(h, iy + half + 1)

    roi = lum[y0:y1, x0:x1]
    roi_h, roi_w = roi.shape

    if roi_h < 5 or roi_w < 5:
        empty_slice = np.zeros(box_size, dtype=np.float32)
        return AnalysisResult(
            fwhm=0.0,
            fwhm_gaussian=0.0,
            fwhm_moffat=0.0,
            sigma=1.0,
            peak=0.0,
            background=0.0,
            background_std=0.0,
            snr=0.0,
            slice=empty_slice,
            slice_fit=empty_slice.copy(),
            slice_x=np.arange(box_size, dtype=np.float32) - half,
            recommended_rl_sigma=1.0,
            centroid_x=cx_init,
            centroid_y=cy_init,
            moffat_alpha=1.5,
            moffat_beta=2.5,
            model="Gaussian",
        )

    # 1. Оценка локального фона и шума по краевым пикселям ROI (с сигма-клиппингом)
    border_mask = np.zeros_like(roi, dtype=bool)
    border_w = max(2, min(roi_h, roi_w) // 6)
    border_mask[:border_w, :] = True
    border_mask[-border_w:, :] = True
    border_mask[:, :border_w] = True
    border_mask[:, -border_w:] = True

    border_pixels = roi[border_mask]
    bg_median = float(np.median(border_pixels))
    mad = float(np.median(np.abs(border_pixels - bg_median)))
    bg_std = max(1.4826 * mad, float(np.std(border_pixels)) if mad == 0.0 else 1.4826 * mad, 1e-6)

    clipped_border = border_pixels[border_pixels <= bg_median + 2.5 * bg_std]
    if clipped_border.size > 0:
        background = float(np.median(clipped_border))
        bg_std = max(float(np.std(clipped_border)), 1e-6)
    else:
        background = bg_median

    # 2. Поиск локального максимума и субпиксельного центроида (итеративный центр масс)
    roi_net = np.maximum(roi - background, 0.0)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

    # Начинаем от самого яркого пикселя в центральной окрестности точки клика
    search_r = max(3.0, half * 0.6)
    dist_click = np.hypot(xx - cx_init, yy - cy_init)
    central_net = np.where(dist_click <= search_r, roi_net, -1.0)
    peak_idx = np.unravel_index(int(np.argmax(central_net)), roi.shape)
    cx = float(xx[peak_idx])
    cy = float(yy[peak_idx])

    centroid_radius = max(4.0, min(half * 0.5, 10.0))
    thresh = 2.0 * bg_std
    for _ in range(3):
        r_grid = np.hypot(xx - cx, yy - cy)
        weights = np.where(r_grid <= centroid_radius, np.maximum(roi_net - thresh, 0.0), 0.0)
        total_w = float(np.sum(weights))
        if total_w <= 1e-7:
            weights = np.where(r_grid <= centroid_radius, roi_net, 0.0)
            total_w = float(np.sum(weights))
        if total_w > 1e-7:
            cx = float(np.sum(xx * weights) / total_w)
            cy = float(np.sum(yy * weights) / total_w)

    # 3. Радиальный профиль и аппроксимация Гауссом / Моффатом
    r_all = np.hypot(xx - cx, yy - cy).ravel()
    z_all = roi_net.ravel()

    fit_mask = r_all <= float(half)
    r_fit = r_all[fit_mask]
    z_fit = z_all[fit_mask]

    peak_raw = float(np.max(roi))
    amp_init = max(float(np.max(z_fit)) if z_fit.size > 0 else 0.0, 1e-4)

    # Начальная оценка сигмы через взвешенный второй момент внутри апертуры
    core_mask = (r_fit <= max(4.0, half * 0.6)) & (z_fit >= 0.15 * amp_init)
    if np.any(core_mask) and float(np.sum(z_fit[core_mask])) > 1e-6:
        var_r = float(np.sum(z_fit[core_mask] * (r_fit[core_mask] ** 2)) / np.sum(z_fit[core_mask]))
        sigma_init = float(np.clip(np.sqrt(max(var_r * 0.5, 0.25)), 0.4, float(half)))
    else:
        sigma_init = 1.5

    # Подгонка 1D радиального Гаусса
    try:
        popt_g, _ = curve_fit(
            _gaussian_1d,
            r_fit,
            z_fit,
            p0=[amp_init, sigma_init],
            bounds=([0.0, 0.25], [max(2.0, amp_init * 2.5), float(half)]),
            maxfev=800
        )
        amp_g, sigma_g = float(popt_g[0]), float(popt_g[1])
        res_g = float(np.mean((z_fit - _gaussian_1d(r_fit, amp_g, sigma_g)) ** 2))
    except Exception:
        amp_g, sigma_g = amp_init, sigma_init
        res_g = float(np.mean((z_fit - _gaussian_1d(r_fit, amp_g, sigma_g)) ** 2))

    fwhm_gaussian = sigma_g * SIGMA_TO_FWHM

    # Подгонка 1D радиального профиля Моффата
    beta_init = 2.5
    alpha_init = max(0.3, fwhm_gaussian / (2.0 * np.sqrt(2.0 ** (1.0 / beta_init) - 1.0)))
    try:
        popt_m, _ = curve_fit(
            _moffat_1d,
            r_fit,
            z_fit,
            p0=[amp_g, alpha_init, beta_init],
            bounds=([0.0, 0.2, 1.1], [max(2.0, amp_init * 2.5), float(half) * 2.0, 12.0]),
            maxfev=1200
        )
        amp_m, alpha_m, beta_m = float(popt_m[0]), float(popt_m[1]), float(popt_m[2])
        fwhm_moffat = float(2.0 * alpha_m * np.sqrt(np.power(2.0, 1.0 / beta_m) - 1.0))
        res_m = float(np.mean((z_fit - _moffat_1d(r_fit, amp_m, alpha_m, beta_m)) ** 2))
    except Exception:
        amp_m, alpha_m, beta_m = amp_g, alpha_init, beta_init
        fwhm_moffat = fwhm_gaussian
        res_m = res_g * 1.1

    # Выбираем лучшую модель по среднеквадратичной ошибке аппроксимации
    use_moffat = (res_m <= res_g * 1.02) and (0.3 <= fwhm_moffat <= box_size * 1.5)
    if use_moffat:
        fwhm = fwhm_moffat
        sigma = fwhm * FWHM_TO_SIGMA
        model_name = "Moffat"
        peak_fitted = min(1.0, background + amp_m)
    else:
        fwhm = fwhm_gaussian
        sigma = sigma_g
        model_name = "Gaussian"
        peak_fitted = min(1.0, background + amp_g)

    peak_val = max(peak_raw, peak_fitted)
    snr = float((peak_val - background) / bg_std) if bg_std > 0 else 0.0

    # 4. Построение 1D профиля сечения (Slice) через субпиксельный центроид
    # Усредняем горизонтальный и вертикальный срезы через (cx, cy) для высокого SNR графика
    rel_coords = np.linspace(-half, half, box_size, dtype=np.float32)
    xs_h = np.clip(cx + rel_coords, 0.0, w - 1.0)
    ys_h = np.full_like(xs_h, np.clip(cy, 0.0, h - 1.0))
    slice_h = map_coordinates(lum, np.vstack((ys_h, xs_h)), order=1, mode='nearest')

    ys_v = np.clip(cy + rel_coords, 0.0, h - 1.0)
    xs_v = np.full_like(ys_v, np.clip(cx, 0.0, w - 1.0))
    slice_v = map_coordinates(lum, np.vstack((ys_v, xs_v)), order=1, mode='nearest')

    slice_profile = (0.5 * (slice_h + slice_v)).astype(np.float32)

    r_slice = np.abs(rel_coords)
    if use_moffat:
        slice_fit = (background + _moffat_1d(r_slice, amp_m, alpha_m, beta_m)).astype(np.float32)
    else:
        slice_fit = (background + _gaussian_1d(r_slice, amp_g, sigma_g)).astype(np.float32)

    # 5. Рекомендуемый Radius (Sigma PSF) для Insert 2 (Richardson-Lucy Deconvolution)
    # Ограничиваем рабочим диапазоном ползунка Insert 2: [0.5, 3.0] px
    recommended_rl_sigma = round(float(np.clip(sigma, 0.5, 3.0)), 2)

    return AnalysisResult(
        fwhm=round(float(fwhm), 3),
        fwhm_gaussian=round(float(fwhm_gaussian), 3),
        fwhm_moffat=round(float(fwhm_moffat), 3),
        sigma=round(float(sigma), 3),
        peak=round(float(peak_val), 4),
        background=round(float(background), 4),
        background_std=round(float(bg_std), 5),
        snr=round(float(snr), 2),
        slice=slice_profile,
        slice_fit=slice_fit,
        slice_x=rel_coords,
        recommended_rl_sigma=recommended_rl_sigma,
        centroid_x=round(float(cx), 2),
        centroid_y=round(float(cy), 2),
        moffat_alpha=round(float(alpha_m), 3),
        moffat_beta=round(float(beta_m), 3),
        model=model_name,
    )


def calculate_angular_scale(
    pixel_distance: float,
    focal_length_mm: float,
    pixel_size_um: float
) -> AnalysisResult:
    """
    Калькулятор углового масштаба и расстояний (Pixel & Angular Distance Calculator).

    Формула масштаба:
        scale_arcsec_per_pixel = (pixel_size_um / focal_length_mm) * 206.265

    Возвращает словарь (AnalysisResult) с дистанцией в пикселях (`pixel_distance`),
    масштабом (`scale_arcsec_per_px`), угловым расстоянием в угловых секундах (`arcsec`),
    угловых минутах (`arcmin`), градусах (`degrees`) и отформатированной строкой (`formatted`).
    """
    px_dist = max(0.0, float(pixel_distance))
    focal_mm = float(focal_length_mm)
    pixel_um = float(pixel_size_um)

    if focal_mm <= 0.0:
        raise ValueError("Фокусное расстояние (focal_length_mm) должно быть больше 0.")
    if pixel_um <= 0.0:
        raise ValueError("Размер пикселя (pixel_size_um) должен быть больше 0.")

    scale_arcsec_per_px = (pixel_um / focal_mm) * ARCSEC_RAD_FACTOR
    arcsec = px_dist * scale_arcsec_per_px
    arcmin = arcsec / 60.0
    degrees = arcsec / 3600.0

    if arcsec >= 60.0:
        formatted = f"{arcsec:.2f}\" ({arcmin:.2f}')"
    else:
        formatted = f"{arcsec:.2f}\""

    return AnalysisResult(
        pixel_distance=round(px_dist, 2),
        scale_arcsec_per_px=round(scale_arcsec_per_px, 4),
        arcsec=round(arcsec, 3),
        arcmin=round(arcmin, 4),
        degrees=round(degrees, 6),
        focal_length_mm=focal_mm,
        pixel_size_um=pixel_um,
        formatted=formatted,
    )
