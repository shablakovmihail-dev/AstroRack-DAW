#"""Математический движок алгоритмов астрономической пост-обработки во float32."""
import cv2
import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter
from scipy.optimize import curve_fit


class AstroEngine:
    """Алгоритмы вейвлетов, деконволюции, выравнивания каналов и тональной компрессии."""

    @staticmethod
    def extract_green_channel(img: np.ndarray) -> np.ndarray:
        """Извлечение зеленого канала (максимальный SNR и резкость на матрицах RGGB)."""
        if img.ndim == 3 and img.shape[2] == 3:
            return img[:, :, 1]
        return img

    @staticmethod
    def auto_rgb_align(img: np.ndarray):
        """Субпиксельное выравнивание R и B каналов относительно G через фазовую корреляцию."""
        if img.ndim != 3 or img.shape[2] != 3:
            return img, (0.0, 0.0), (0.0, 0.0)

        h, w = img.shape[:2]
        r = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 2].astype(np.float32)

        hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx_r, dy_r), _ = cv2.phaseCorrelate(g, r, hann)
        (dx_b, dy_b), _ = cv2.phaseCorrelate(g, b, hann)

        m_r = np.array([[1.0, 0.0, -dx_r], [0.0, 1.0, -dy_r]], dtype=np.float32)
        m_b = np.array([[1.0, 0.0, -dx_b], [0.0, 1.0, -dy_b]], dtype=np.float32)

        r_aligned = cv2.warpAffine(r, m_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        b_aligned = cv2.warpAffine(b, m_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        aligned_img = np.stack([r_aligned, g, b_aligned], axis=-1)
        return np.clip(aligned_img, 0.0, 1.0), (-dx_r, -dy_r), (-dx_b, -dy_b)

    # --- Вейвлеты à trous (B3-сплайн) с динамическим числом слоёв ---------------------------
    WAVELET_MIN_LAYERS = 1
    WAVELET_MAX_LAYERS = 6
    WAVELET_DENOISE_SCALE = 0.035
    _B3_KERNEL = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0

    @staticmethod
    def normalize_layers_config(layers_config=None, legacy_l1: float = 1.0, legacy_l2: float = 1.0,
                                legacy_denoise: float = 0.0) -> list:
        """Приводит конфиг слоёв к списку словарей {"amount", "denoise", "mode"} длиной 1..6.

        Принимает список словарей или кортежей (amount[, denoise[, mode]]). Если конфиг не передан,
        собирается из старых параметров пресета (w_l1, w_l2, w_denoise): старый глобальный
        denoise относился только к Layer 1, поэтому Layer 2 получает denoise=0.
        """
        if not layers_config:
            layers_config = [
                {"amount": legacy_l1, "denoise": legacy_denoise},
                {"amount": legacy_l2, "denoise": 0.0},
            ]

        norm = []
        for item in layers_config:
            if isinstance(item, dict):
                amount = item.get("amount", item.get("gain", 1.0))
                denoise = item.get("denoise", 0.0)
                mode = item.get("mode", "soft")
            else:
                seq = list(item)
                amount = seq[0] if len(seq) > 0 else 1.0
                denoise = seq[1] if len(seq) > 1 else 0.0
                mode = seq[2] if len(seq) > 2 else "soft"
            norm.append({
                "amount": round(float(amount), 4),
                "denoise": round(max(0.0, float(denoise)), 4),
                "mode": "hard" if str(mode).lower() == "hard" else "soft",
            })

        return norm[:AstroEngine.WAVELET_MAX_LAYERS]

    @staticmethod
    def trim_neutral_layers(layers: list) -> list:
        """Отбрасывает хвостовые нейтральные слои (amount=1, denoise=0): они эквивалентно остаются в остатке."""
        n = len(layers)
        while n > AstroEngine.WAVELET_MIN_LAYERS and abs(layers[n - 1]["amount"] - 1.0) < 1e-4 \
                and layers[n - 1]["denoise"] <= 1e-5:
            n -= 1
        return layers[:n]

    @staticmethod
    def atrous_decompose(img: np.ndarray, n_layers: int, cancel_check=None):
        """B3-разложение à trous на n_layers (1..6) уровней деталей.

        Возвращает (residual, [w1, ..., wn]); img == residual + sum(w_k). Работает для 2D и HxWxC.
        cancel_check — необязательный callable, возвращающий True для досрочной отмены (тогда None).
        """
        n_layers = int(np.clip(n_layers, AstroEngine.WAVELET_MIN_LAYERS, AstroEngine.WAVELET_MAX_LAYERS))
        base = AstroEngine._B3_KERNEL
        details = []
        current = img
        for level in range(n_layers):
            step = 2 ** level
            if step == 1:
                kernel = base
            else:
                kernel = np.zeros(4 * step + 1, dtype=np.float32)
                kernel[0::step] = base

            smooth = convolve1d(current, kernel, axis=0, mode='reflect')
            if cancel_check is not None and cancel_check():
                return None
            smooth = convolve1d(smooth, kernel, axis=1, mode='reflect')
            if cancel_check is not None and cancel_check():
                return None

            details.append(current - smooth)
            current = smooth

        return current, details

    @staticmethod
    def atrous_recompose(residual: np.ndarray, details: list, layers: list) -> np.ndarray:
        """Сборка: residual + sum(amount_k * threshold_k(w_k)). Порог пропускается при denoise == 0."""
        out = residual.astype(np.float32, copy=True)
        for w, cfg in zip(details, layers):
            if cfg["denoise"] > 0.0:
                thresh = np.float32(cfg["denoise"] * AstroEngine.WAVELET_DENOISE_SCALE)
                if cfg["mode"] == "hard":
                    w = np.where(np.abs(w) >= thresh, w, np.float32(0.0))
                else:
                    w = np.sign(w) * np.maximum(np.abs(w) - thresh, np.float32(0.0))
            amount = cfg["amount"]
            if abs(amount - 1.0) < 1e-6:
                out += w
            else:
                out += w * np.float32(amount)
        return out

    @staticmethod
    def wavelet_sharpening(img: np.ndarray, layer1_gain: float = 1.0, layer2_gain: float = 1.0,
                           denoise: float = 0.0, layers_config=None) -> np.ndarray:
        """Вейвлеты à trous (B3-сплайн), 1..6 слоёв. 1.00 — нейтраль, >1.00 — усиление деталей.

        Если layers_config не задан, используется старый формат из 2 слоёв (layer1_gain, layer2_gain, denoise).
        """
        layers = AstroEngine.normalize_layers_config(layers_config, layer1_gain, layer2_gain, denoise)
        layers = AstroEngine.trim_neutral_layers(layers)
        residual, details = AstroEngine.atrous_decompose(img.astype(np.float32, copy=False), len(layers))
        out = AstroEngine.atrous_recompose(residual, details, layers)
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def unsharp_mask(img: np.ndarray, amount: float, radius: float, threshold: float) -> np.ndarray:
        """Нерезкое маскирование с порогом (Thresholding) для отсечения фонового шума."""
        def unsharp_plane(plane: np.ndarray) -> np.ndarray:
            blurred = gaussian_filter(plane, sigma=radius, mode='reflect')
            diff = plane - blurred
            if threshold > 0.0:
                diff_g = np.sign(diff) * np.maximum(np.abs(diff) - threshold, 0.0)
            else:
                diff_g = diff
            return plane + (amount - 1.0) * diff_g

        if img.ndim == 2:
            out = unsharp_plane(img)
        else:
            planes = [unsharp_plane(img[:, :, c]) for c in range(img.shape[2])]
            out = np.stack(planes, axis=-1)

        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def richardson_lucy(img: np.ndarray, sigma: float, iterations: int, protect_edge: bool = False) -> np.ndarray:
        """Деконволюция Ричардсона-Люси с защитой края лимба (Anti-Ringing)."""
        if iterations <= 0:
            return img

        def rl_plane(plane: np.ndarray) -> np.ndarray:
            estimate = np.maximum(plane.copy(), 1e-6)
            for _ in range(iterations):
                conv = gaussian_filter(estimate, sigma=sigma, mode='reflect')
                conv = np.maximum(conv, 1e-12)
                relative_blur = plane / conv
                corr = gaussian_filter(relative_blur, sigma=sigma, mode='reflect')
                estimate = estimate * corr

            if protect_edge:
                gx = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
                grad = np.sqrt(gx**2 + gy**2)
                edge_mask = np.clip(gaussian_filter(grad, sigma=2.0) * 4.0, 0.0, 1.0)
                estimate = estimate * (1.0 - edge_mask * 0.85) + plane * (edge_mask * 0.85)

            return estimate

        if img.ndim == 2:
            out = rl_plane(img)
        else:
            planes = [rl_plane(img[:, :, c]) for c in range(img.shape[2])]
            out = np.stack(planes, axis=-1)

        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def tone_and_contrast(img: np.ndarray, black_level: float, gamma: float, contrast: float) -> np.ndarray:
        """Отсечка фона, гамма средних тонов и параметрическая S-Curve контраста."""
        denom = max(1.0 - black_level, 1e-4)
        out = np.clip((img - black_level) / denom, 0.0, 1.0)

        if abs(gamma - 1.0) > 1e-3 and gamma > 0.0:
            out = np.power(out, 1.0 / gamma)

        if abs(contrast - 1.0) > 1e-3:
            eps = 1e-7
            out_clamped = np.clip(out, eps, 1.0 - eps)
            k = contrast
            out = (out_clamped ** k) / ((out_clamped ** k) + ((1.0 - out_clamped) ** k))

        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def rotate_90_cw(img: np.ndarray) -> np.ndarray:
        """Поворот кадра на 90° по часовой стрелке (оси Y, X)."""
        return np.ascontiguousarray(np.rot90(img, k=-1))

    @staticmethod
    def rotate_90_ccw(img: np.ndarray) -> np.ndarray:
        """Поворот кадра на 90° против часовой стрелки (оси Y, X)."""
        return np.ascontiguousarray(np.rot90(img, k=1))

    @staticmethod
    def flip_horizontal(img: np.ndarray) -> np.ndarray:
        """Зеркальное отражение по горизонтали (ось X)."""
        return np.ascontiguousarray(np.fliplr(img))

    @staticmethod
    def flip_vertical(img: np.ndarray) -> np.ndarray:
        """Зеркальное отражение по вертикали (ось Y)."""
        return np.ascontiguousarray(np.flipud(img))
    @staticmethod
    def compute_psf_fwhm(img: np.ndarray, x: float, y: float, box_radius: int = 14) -> dict:
        """
        Расчет Full Width at Half Maximum (FWHM) и параметров Гауссовой PSF звезды/кратера.
        Возвращает метрики FWHM, Sigma, Peak, Background, а также профиль для графика.
        """
        if img is None:
            return None

        if img.ndim == 3:
            lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            lum = img

        h, w = lum.shape[:2]
        ix = int(round(x))
        iy = int(round(y))

        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return None

        x1 = max(0, ix - box_radius)
        x2 = min(w, ix + box_radius + 1)
        y1 = max(0, iy - box_radius)
        y2 = min(h, iy + box_radius + 1)

        patch = lum[y1:y2, x1:x2].astype(np.float32)
        if patch.shape[0] < 5 or patch.shape[1] < 5:
            return None

        # Локальный максимум в окрестности клика
        py, px = np.unravel_index(np.argmax(patch), patch.shape)
        peak_val = float(patch[py, px])

        # Оценка фона по внешнему кольцу пикселей
        border_pixels = np.concatenate([
            patch[0, :], patch[-1, :],
            patch[:, 0], patch[:, -1]
        ])
        bg_val = float(np.median(border_pixels))
        amp = max(1e-4, peak_val - bg_val)

        # Центроид около пика (размер окна 7x7)
        w_r = 3
        wx1 = max(0, px - w_r)
        wx2 = min(patch.shape[1], px + w_r + 1)
        wy1 = max(0, py - w_r)
        wy2 = min(patch.shape[0], py + w_r + 1)
        w_patch = np.maximum(patch[wy1:wy2, wx1:wx2] - bg_val, 0)
        total_w = np.sum(w_patch)

        if total_w > 1e-6:
            yy, xx = np.indices(w_patch.shape)
            sub_x = float(wx1 + np.sum(xx * w_patch) / total_w)
            sub_y = float(wy1 + np.sum(yy * w_patch) / total_w)
        else:
            sub_x = float(px)
            sub_y = float(py)

        abs_xc = x1 + sub_x
        abs_yc = y1 + sub_y

        # Радиальные расстояния от центра масс
        gy, gx = np.indices(patch.shape)
        r_mat = np.sqrt((gx - sub_x) ** 2 + (gy - sub_y) ** 2).flatten()
        val_vec = patch.flatten()

        # Моментная оценка сигмы
        weights = np.maximum(val_vec - bg_val, 0)
        w_sum = np.sum(weights)
        if w_sum > 1e-6:
            sigma_moment = float(np.sqrt(max(0.1, np.sum(weights * (r_mat ** 2)) / (2.0 * w_sum))))
        else:
            sigma_moment = 1.5

        sigma_fit = sigma_moment
        try:
            def gauss_func(r, a, s, b):
                return b + a * np.exp(-0.5 * (r / np.maximum(s, 0.2)) ** 2)

            popt, _ = curve_fit(
                gauss_func, r_mat, val_vec,
                p0=[amp, sigma_moment, bg_val],
                bounds=([0.0, 0.2, 0.0], [2.0, 20.0, 1.0]),
                maxfev=600
            )
            amp = float(popt[0])
            sigma_fit = float(popt[1])
            bg_val = float(popt[2])
        except Exception:
            pass

        sigma = float(np.clip(sigma_fit, 0.3, 15.0))
        fwhm = 2.35482 * sigma

        # Горизонтальное сечение через центр для графика
        cut_r = min(12, int(patch.shape[1] // 2))
        xs = np.linspace(-cut_r, cut_r, 41)
        row_idx = int(np.clip(round(sub_y), 0, patch.shape[0] - 1))
        col_coords = np.clip(np.round(sub_x + xs).astype(int), 0, patch.shape[1] - 1)
        data_cut = patch[row_idx, col_coords]
        model_cut = bg_val + amp * np.exp(-0.5 * (xs / sigma) ** 2)

        return {
            'fwhm': round(fwhm, 2),
            'sigma': round(sigma, 2),
            'peak': round(peak_val, 3),
            'background': round(bg_val, 3),
            'amplitude': round(amp, 3),
            'center_scene': (round(abs_xc, 1), round(abs_yc, 1)),
            'xs': xs.tolist(),
            'data_cut': data_cut.tolist(),
            'model_cut': model_cut.tolist()
        }

    # --- Новые алгоритмы: Local Contrast / Clarity и Color Engine -------------------
    @staticmethod
    def apply_local_contrast(image: np.ndarray, amount: float = 2.0, radius: float = 16.0,
                             method: str = 'clahe') -> np.ndarray:
        """Локальный контраст / Clarity (CLAHE по L или Gaussian High-Pass)."""
        from .enhance import apply_local_contrast as _apply_lc
        return _apply_lc(image, amount=amount, radius=radius, method=method)

    @staticmethod
    def apply_color_engine(image: np.ndarray, saturation: float = 1.0, mineral_boost: float = 0.0,
                           bg_neutralize: bool = False, bg_coords=None) -> np.ndarray:
        """Color Engine: насыщенность, Mineral Moon Boost и нейтрализация космического фона."""
        from .enhance import apply_color_engine as _apply_ce
        return _apply_ce(image, saturation=saturation, mineral_boost=mineral_boost,
                         bg_neutralize=bg_neutralize, bg_coords=bg_coords)

    @staticmethod
    def blend_dry_wet(dry_img: np.ndarray, wet_img: np.ndarray, wet_ratio: float = 1.0) -> np.ndarray:
        """Универсальное параллельное подмешивание Dry/Wet (float32/uint8)."""
        from .pipeline import blend_dry_wet as _blend
        return _blend(dry_img, wet_img, wet_ratio)

    @staticmethod
    def soft_clip_limiter(img: np.ndarray, threshold: float = 0.88) -> np.ndarray:
        """Мастер-лимитер светов (Master Soft-Clip Limiter) с гладким тангенциальным коленом."""
        return soft_clip_limiter(img, threshold)


def soft_clip_limiter(img: np.ndarray, threshold: float = 0.88) -> np.ndarray:
    """Мастер-лимитер светов (Master Soft-Clip Limiter).

    Мягкое сжатие пересветов гладким тангенциальным коленом (Soft-Knee Knee Roll-off):
      y = x,                                           при x <= T
      y = T + (1.0 - T) * np.tanh((x - T) / (1.0 - T)), при x > T

    Работает в нормализованном диапазоне [0.0, 1.0] с монохромными (2D)
    и цветными (3D) изображениями float32. Предотвращает жесткую отсечку (clipping),
    сохраняя микротекстуру деталей в ярких зонах планет и лунных кратеров.
    """
    if img is None or img.size == 0:
        return img

    out = np.array(img, dtype=np.float32, copy=True)
    t = float(threshold)
    scale = 1.0 - t
    if scale <= 1e-6:
        return np.clip(out, 0.0, 1.0)

    over = out > t
    if np.any(over):
        diff = (out[over] - t) / scale
        out[over] = t + scale * np.tanh(diff)

    return np.clip(out, 0.0, 1.0)

