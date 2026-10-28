#"""Фоновый поток (Worker QThread) обработки кадров с отменой задач и кэшированием стадий."""
import threading
import cv2
import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from .processing import AstroEngine
from .enhance import apply_local_contrast, apply_color_engine


def blend_dry_wet(dry_img: np.ndarray, wet_img: np.ndarray, wet_ratio: float = 1.0) -> np.ndarray:
    """Универсальная быстрая функция параллельного линейного подмешивания Dry/Wet (float32/uint8).

    Параметры:
        dry_img: входной необработанный массив (Dry).
        wet_img: обработанный массив инсерта (Wet).
        wet_ratio: коэффициент от 0.0 (100% dry) до 1.0 (100% wet), либо в процентах 0..100.

    Возвращает:
        Результирующий массив с той же размерностью и dtype, что и wet_img.
    """
    if dry_img is None:
        return wet_img
    if wet_img is None:
        return dry_img

    ratio = float(wet_ratio)
    if ratio > 1.0:
        ratio /= 100.0
    ratio = max(0.0, min(1.0, ratio))

    if ratio >= 0.999:
        return wet_img
    if ratio <= 0.001:
        return dry_img

    dry_c = dry_img
    wet_c = wet_img

    # Согласование типов данных (cv2.addWeighted требует одинаковый dtype)
    if dry_c.dtype != wet_c.dtype:
        dry_c = dry_c.astype(wet_c.dtype)

    # Согласование размерностей (ч/б 2D vs цветной 3D)
    if dry_c.ndim == 2 and wet_c.ndim == 3:
        dry_c = cv2.cvtColor(dry_c, cv2.COLOR_GRAY2BGR)
    elif dry_c.ndim == 3 and wet_c.ndim == 2:
        wet_c = cv2.cvtColor(wet_c, cv2.COLOR_GRAY2BGR)

    # Защита от несовпадения пространственных размеров
    if dry_c.shape[:2] != wet_c.shape[:2]:
        dry_c = cv2.resize(dry_c, (wet_c.shape[1], wet_c.shape[0]))

    # Гарантия непрерывности в памяти для cv2.addWeighted
    if not wet_c.flags['C_CONTIGUOUS']:
        wet_c = np.ascontiguousarray(wet_c)
    if not dry_c.flags['C_CONTIGUOUS']:
        dry_c = np.ascontiguousarray(dry_c)

    return cv2.addWeighted(wet_c, float(ratio), dry_c, float(1.0 - ratio), 0.0)


def _extract_wet_ratio(params: dict, *keys, default: float = 1.0) -> float:
    """Извлекает коэффициент wet_ratio (0.0..1.0) из параметров по списку возможных ключей."""
    if not params:
        return float(default)
    for k in keys:
        if k in params and params[k] is not None:
            val = float(params[k])
            if val > 1.0:
                val /= 100.0
            return max(0.0, min(1.0, val))
    return float(default)


class PipelineWorker(QObject):
    """Worker, выполняющий расчет цепочки инсертов в фоновом потоке с поэтапным кэшированием."""
    frame_ready = pyqtSignal(np.ndarray)
    rgb_shifts_computed = pyqtSignal(float, float, float, float)
    _wake_worker = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._raw_image = None
        self._pending_params = None
        self._is_busy = False
        self._task_id = 0

        # Поэтапные кэши цепочки обработки
        self._clear_caches()

        # Гарантированный запуск слота в фоновом потоке QThread через QueuedConnection
        self._wake_worker.connect(self._process_loop, Qt.ConnectionType.QueuedConnection)

    def _clear_caches(self):
        """Сброс всех промежуточных кэшей при смене исходного изображения."""
        self._cache_rgb_align = None          # (aligned_img, (dx_r, dy_r, dx_b, dy_b))
        self._cache_stage4 = {}               # key4 -> img_after_insert4

        self._cache_wavelet_decomp = None     # (key4, c2, w1, w2)
        self._cache_unsharp_blur = None       # (key4, radius, blurred_img)
        self._cache_stage1_wet = None         # (core_key1, wet_img)
        self._cache_stage1 = None             # (key1, img_after_insert1)
        self._cache_stage1_active = None      # (active_key1, img_after_insert1)

        self._cache_rl_core = None            # (key1, sigma, iters, rl_estimate)
        self._cache_rl_edge_mask = None       # (key1, edge_mask)
        self._cache_stage2_wet = None         # (core_key2, wet_img)
        self._cache_stage2 = None             # (key2, img_after_insert2)
        self._cache_stage2_active = None      # (active_key2, img_after_insert2)

        self._cache_stage3 = None             # (key3, img_after_insert3)
        self._cache_stage3_active = None      # (active_key3, img_after_insert3)

        self._stage_cache = {}                # slot -> (key, img): Insert 5 / Insert 6

    def cancel_current(self):
        """Немедленно помечает текущий расчет как устаревший."""
        with self._lock:
            self._task_id += 1
            self._pending_params = None

    def set_source_image(self, img: np.ndarray):
        """Устанавливает новый исходный кадр и сбрасывает кэш стадий (потокобезопасно)."""
        if img is None or img.size == 0:
            with self._lock:
                self._task_id += 1
                self._raw_image = None
                self._pending_params = None
                self._clear_caches()
            return

        raw_copy = np.ascontiguousarray(img, dtype=np.float32).copy()
        with self._lock:
            self._task_id += 1
            self._raw_image = raw_copy
            self._pending_params = None
            self._clear_caches()

    def request_processing(self, params: dict):
        """Ставит новый запрос в очередь с отменой предыдущего незавершенного расчета."""
        if params is None:
            return
        with self._lock:
            self._task_id += 1
            self._pending_params = params.copy()
            should_wake = not self._is_busy
        if should_wake:
            self._wake_worker.emit()

    def update_insert_param(self, insert_id: int, param_name: str, value):
        """Прямое точечное обновление параметра инсерта в очереди пайплайна."""
        val = float(value)
        if val > 1.0:
            val /= 100.0
        with self._lock:
            if self._pending_params is None:
                self._pending_params = {}
            self._pending_params[f"insert{insert_id}_{param_name}"] = val
            self._task_id += 1
            should_wake = not self._is_busy
        if should_wake:
            self._wake_worker.emit()

    def _is_cancelled(self, task_id: int) -> bool:
        return self._task_id != task_id

    def _process_loop(self):
        """Основной цикл обработки в фоновом потоке QThread."""
        while True:
            with self._lock:
                if self._raw_image is None or self._pending_params is None:
                    self._is_busy = False
                    return
                self._is_busy = True
                params = self._pending_params
                self._pending_params = None
                task_id = self._task_id
                raw_img = self._raw_image

            result = self._execute_pipeline(raw_img, params, task_id)
            if result is not None and not self._is_cancelled(task_id):
                self.frame_ready.emit(result)

    def _execute_pipeline(self, raw_img: np.ndarray, params: dict, task_id: int):
        if raw_img is None or raw_img.size == 0 or self._is_cancelled(task_id):
            return None

        # Master Bypass: мгновенный возврат исходного кадра без сброса кэшей инсертов
        if params.get('global_bypass', False):
            return raw_img

        # =====================================================================
        # СТАДИЯ 1 (Insert 4): Auto RGB Align & Extract Green Channel
        # =====================================================================
        ins4_on = bool(params.get('insert4_enabled', False))
        do_align = ins4_on and bool(params.get('auto_rgb_align', False))
        do_green = ins4_on and bool(params.get('extract_green', False))
        key4 = (do_align, do_green)

        if key4 in self._cache_stage4:
            stage4_img = self._cache_stage4[key4]
            if do_align and self._cache_rgb_align is not None:
                dx_r, dy_r, dx_b, dy_b = self._cache_rgb_align[1]
                self.rgb_shifts_computed.emit(dx_r, dy_r, dx_b, dy_b)
        else:
            stage4_img = raw_img
            if do_align:
                if self._cache_rgb_align is not None:
                    stage4_img, shifts = self._cache_rgb_align
                else:
                    if self._is_cancelled(task_id):
                        return None
                    aligned, (dx_r, dy_r), (dx_b, dy_b) = AstroEngine.auto_rgb_align(stage4_img)
                    stage4_img = np.ascontiguousarray(aligned, dtype=np.float32)
                    shifts = (dx_r, dy_r, dx_b, dy_b)
                    self._cache_rgb_align = (stage4_img, shifts)
                self.rgb_shifts_computed.emit(*shifts)

            if self._is_cancelled(task_id):
                return None

            if do_green:
                stage4_img = np.ascontiguousarray(
                    AstroEngine.extract_green_channel(stage4_img), dtype=np.float32
                )

            self._cache_stage4[key4] = stage4_img

        if self._is_cancelled(task_id):
            return None

        # =====================================================================
        # СТАДИЯ 2 (Insert 1): Wavelet (à trous) / Unsharp Mask
        # =====================================================================
        ins1_on = bool(params.get('insert1_enabled', False))
        w_mode = int(params.get('w_mode', 0))
        w_l1 = round(float(params.get('w_l1', 1.0)), 4)
        w_l2 = round(float(params.get('w_l2', 1.0)), 4)
        w_denoise = round(float(params.get('w_denoise', 0.0)), 4)
        w_wet_ratio = _extract_wet_ratio(params, 'w_wet_mix', 'insert1_wet_mix', 'wet_mix', default=1.0)

        # Динамический конфиг слоёв: params['layers_config'] или старые w_l1/w_l2/w_denoise (2 слоя)
        w_layers = None
        if w_mode == 0:
            w_layers = AstroEngine.trim_neutral_layers(
                AstroEngine.normalize_layers_config(
                    params.get('layers_config'), w_l1, w_l2, w_denoise
                )
            )

        if w_mode == 0:
            ins1_neutral = not ins1_on or (
                len(w_layers) == 1 and abs(w_layers[0]["amount"] - 1.0) < 1e-4 and w_layers[0]["denoise"] <= 1e-5
            )
        else:
            ins1_neutral = not ins1_on or abs(w_l1 - 1.0) < 1e-4

        if ins1_neutral or w_wet_ratio <= 0.001:
            key1 = (key4, 'bypass')
            stage1_img = stage4_img
        else:
            if w_mode == 0:
                core_key1 = (key4, w_mode, tuple((l["amount"], l["denoise"], l["mode"]) for l in w_layers))
            else:
                core_key1 = (key4, w_mode, w_l1, w_l2, w_denoise)

            key1 = (core_key1, round(w_wet_ratio, 4))

            if self._cache_stage1 is not None and self._cache_stage1[0] == key1:
                stage1_img = self._cache_stage1[1]
            elif self._cache_stage1_active is not None and self._cache_stage1_active[0] == key1:
                stage1_img = self._cache_stage1_active[1]
            else:
                # Извлечение или вычисление неразбавленного wet-изображения
                if self._cache_stage1_wet is not None and self._cache_stage1_wet[0] == core_key1:
                    stage1_wet = self._cache_stage1_wet[1]
                else:
                    stage1_wet = self._compute_stage1(stage4_img, key4, w_mode, w_l1, w_l2, w_denoise, task_id,
                                                      layers=w_layers)
                    if stage1_wet is None:
                        return None
                    self._cache_stage1_wet = (core_key1, stage1_wet)

                # Параллельное подмешивание Dry/Wet
                stage1_img = blend_dry_wet(stage4_img, stage1_wet, w_wet_ratio)
                self._cache_stage1_active = (key1, stage1_img)

        self._cache_stage1 = (key1, stage1_img)

        if self._is_cancelled(task_id):
            return None

        # =====================================================================
        # СТАДИЯ 3 (Insert 2): Richardson-Lucy Deconvolution
        # =====================================================================
        ins2_on = bool(params.get('insert2_enabled', False))
        rl_sigma = round(float(params.get('rl_sigma', 1.0)), 3)
        rl_iters = int(params.get('rl_iters', 0))
        rl_anti_ringing = bool(params.get('rl_anti_ringing', False))
        rl_wet_ratio = _extract_wet_ratio(params, 'rl_wet_mix', 'insert2_wet_mix', 'wet_mix', default=1.0)
        is_preview = bool(params.get('_preview', False))

        if not ins2_on or rl_iters <= 0 or rl_wet_ratio <= 0.001:
            key2 = (key1, 'bypass')
            stage2_img = stage1_img
        else:
            full_core_key2 = (key1, rl_sigma, rl_iters, rl_anti_ringing)
            eff_iters = min(rl_iters, 5) if is_preview else rl_iters
            core_key2 = (key1, rl_sigma, eff_iters, rl_anti_ringing)

            key2 = (core_key2, round(rl_wet_ratio, 4))
            full_key2 = (full_core_key2, round(rl_wet_ratio, 4))

            # Если уже есть точный кэш для полного числа итераций с данным wet_mix
            if self._cache_stage2 is not None and self._cache_stage2[0] in (key2, full_key2):
                stage2_img = self._cache_stage2[1]
            elif self._cache_stage2_active is not None and self._cache_stage2_active[0] == full_key2:
                stage2_img = self._cache_stage2_active[1]
            else:
                if self._cache_stage2_wet is not None and self._cache_stage2_wet[0] == full_core_key2:
                    stage2_wet = self._cache_stage2_wet[1]
                elif self._cache_stage2_wet is not None and self._cache_stage2_wet[0] == core_key2:
                    stage2_wet = self._cache_stage2_wet[1]
                else:
                    stage2_wet = self._compute_stage2(
                        stage1_img, key1, rl_sigma, eff_iters, rl_anti_ringing, task_id
                    )
                    if stage2_wet is None:
                        return None
                    if not is_preview or eff_iters == rl_iters:
                        self._cache_stage2_wet = (full_core_key2, stage2_wet)
                    else:
                        self._cache_stage2_wet = (core_key2, stage2_wet)

                # Параллельное подмешивание Dry/Wet
                stage2_img = blend_dry_wet(stage1_img, stage2_wet, rl_wet_ratio)
                if not is_preview or eff_iters == rl_iters:
                    self._cache_stage2_active = (full_key2, stage2_img)

        self._cache_stage2 = (key2, stage2_img)

        if self._is_cancelled(task_id):
            return None

        # =====================================================================
        # СТАДИЯ 4 (Insert 3): Tone & Contrast (Curves) [Исключен из Dry/Wet]
        # =====================================================================
        ins3_on = bool(params.get('insert3_enabled', False))
        tc_black = round(float(params.get('tc_black', 0.0)), 4)
        tc_gamma = round(float(params.get('tc_gamma', 1.0)), 4)
        tc_contrast = round(float(params.get('tc_contrast', 1.0)), 4)

        ins3_neutral = (
            not ins3_on
            or (tc_black <= 1e-5 and abs(tc_gamma - 1.0) <= 1e-3 and abs(tc_contrast - 1.0) <= 1e-3)
        )

        if ins3_neutral:
            key3 = (key2, 'bypass')
            stage3_img = stage2_img
        else:
            key3 = (key2, tc_black, tc_gamma, tc_contrast, is_preview)
            full_key3 = (key2, tc_black, tc_gamma, tc_contrast, False)
            if self._cache_stage3 is not None and self._cache_stage3[0] in (key3, full_key3):
                stage3_img = self._cache_stage3[1]
            elif self._cache_stage3_active is not None and self._cache_stage3_active[0] == full_key3:
                stage3_img = self._cache_stage3_active[1]
            else:
                stage3_img = self._compute_stage3(
                    stage2_img, tc_black, tc_gamma, tc_contrast, is_preview, task_id
                )
                if stage3_img is None:
                    return None
                if not is_preview:
                    self._cache_stage3_active = (full_key3, stage3_img)

        self._cache_stage3 = (key3, stage3_img)

        if self._is_cancelled(task_id):
            return None

        # =====================================================================
        # СТАДИЯ 5 (Insert 5): Local Contrast / Clarity (CLAHE / High-Pass)
        # =====================================================================
        ins5_on = bool(params.get('insert5_enabled', False))
        lc_method = str(params.get('lc_method', 'clahe')).lower()
        lc_is_hp = lc_method in ('high_pass', 'highpass', 'hp')
        lc_amount = round(float(params.get('lc_amount', 0.0 if lc_is_hp else 1.0)), 3)
        lc_radius = round(float(params.get('lc_radius', 30.0 if lc_is_hp else 16.0)), 2)
        lc_wet_ratio = _extract_wet_ratio(params, 'lc_wet_mix', 'insert5_wet_mix', 'wet_mix', default=1.0)

        ins5_neutral = (not ins5_on) or (lc_amount <= 1e-3 if lc_is_hp else lc_amount <= 1.0 + 1e-3)
        if ins5_neutral or lc_wet_ratio <= 0.001:
            key5 = (key3, 'bypass')
            stage5_img = stage3_img
        else:
            core_key5 = (key3, 'lc', lc_is_hp, lc_amount, lc_radius)
            key5 = (core_key5, round(lc_wet_ratio, 4))

            cached5 = self._stage_cache.get('stage5')
            if cached5 is not None and cached5[0] == key5:
                stage5_img = cached5[1]
            else:
                stage5_wet = self._run_cached_stage(
                    'stage5_wet', core_key5,
                    lambda: apply_local_contrast(
                        stage3_img, lc_amount, lc_radius, 'high_pass' if lc_is_hp else 'clahe'
                    ),
                    task_id
                )
                if stage5_wet is None:
                    return None
                # Параллельное подмешивание Dry/Wet
                stage5_img = blend_dry_wet(stage3_img, stage5_wet, lc_wet_ratio)
                self._stage_cache['stage5'] = (key5, stage5_img)

        # =====================================================================
        # СТАДИЯ 6 (Insert 6): Color Engine & Background Neutralization
        # =====================================================================
        ins6_on = bool(params.get('insert6_enabled', False))
        cs_sat = round(float(params.get('cs_saturation', 1.0)), 3)
        cs_mineral = round(float(params.get('cs_mineral', 0.0)), 3)
        cs_bg = bool(params.get('cs_bg_neutralize', False))
        cs_coords = params.get('cs_bg_coords', None)
        cs_coords = tuple(float(v) for v in cs_coords) if cs_coords is not None else None
        cs_wet_ratio = _extract_wet_ratio(params, 'cs_wet_mix', 'insert6_wet_mix', 'wet_mix', default=1.0)

        ins6_neutral = (
            (not ins6_on)
            or stage5_img.ndim != 3 or stage5_img.shape[2] < 3          # ч/б: нечего обрабатывать
            or (abs(cs_sat - 1.0) <= 1e-3 and cs_mineral <= 1e-3 and not cs_bg)
        )
        if ins6_neutral or cs_wet_ratio <= 0.001:
            return stage5_img

        if self._is_cancelled(task_id):
            return None

        core_key6 = (key5, 'cs', cs_sat, cs_mineral, cs_bg, cs_coords)
        key6 = (core_key6, round(cs_wet_ratio, 4))

        cached6 = self._stage_cache.get('stage6')
        if cached6 is not None and cached6[0] == key6:
            return cached6[1]

        stage6_wet = self._run_cached_stage(
            'stage6_wet', core_key6,
            lambda: apply_color_engine(stage5_img, cs_sat, cs_mineral, cs_bg, cs_coords),
            task_id
        )
        if stage6_wet is None:
            return None
        stage6_img = blend_dry_wet(stage5_img, stage6_wet, cs_wet_ratio)
        self._stage_cache['stage6'] = (key6, stage6_img)
        return stage6_img

    def _run_cached_stage(self, slot: str, key, compute_fn, task_id: int):
        """Одноэлементный кэш стадии: пересчёт только при смене ключа (вход + параметры).

        Слот хранит последний реально вычисленный результат, поэтому переключение
        байпаса инсерта вперёд-назад не приводит к повторному расчёту.
        """
        cached = self._stage_cache.get(slot)
        if cached is not None and cached[0] == key:
            return cached[1]
        if self._is_cancelled(task_id):
            return None
        result = compute_fn()
        if self._is_cancelled(task_id):
            return None
        self._stage_cache[slot] = (key, result)
        return result

    def _compute_stage1(self, img: np.ndarray, key4, w_mode: int,
                        w_l1: float, w_l2: float, w_denoise: float, task_id: int, layers=None):
        """Вейвлеты (1..6 слоёв) / Unsharp Mask с кэшированием декомпозиции слоев."""
        if w_mode == 0:
            if layers is None:
                layers = AstroEngine.trim_neutral_layers(
                    AstroEngine.normalize_layers_config(None, w_l1, w_l2, w_denoise)
                )
            n_layers = len(layers)

            # Кэш декомпозиции зависит только от входа и числа слоёв (не от amount/denoise)
            decomp_key = (key4, n_layers)
            if self._cache_wavelet_decomp is not None and self._cache_wavelet_decomp[0] == decomp_key:
                _, residual, details = self._cache_wavelet_decomp
            else:
                res = AstroEngine.atrous_decompose(
                    img, n_layers, cancel_check=lambda: self._is_cancelled(task_id)
                )
                if res is None:
                    return None
                residual, details = res
                self._cache_wavelet_decomp = (decomp_key, residual, details)

            if self._is_cancelled(task_id):
                return None

            out = AstroEngine.atrous_recompose(residual, details, layers)
            return np.clip(out, 0.0, 1.0, out=out)

        else:
            # Unsharp Mask: кэшируем размытую копию по (key4, radius)
            blur_key = (key4, w_l2)
            if self._cache_unsharp_blur is not None and self._cache_unsharp_blur[0] == blur_key:
                blurred = self._cache_unsharp_blur[1]
            else:
                if img.ndim == 2:
                    blurred = gaussian_filter(img, sigma=w_l2, mode='reflect')
                else:
                    planes = []
                    for c in range(img.shape[2]):
                        if self._is_cancelled(task_id):
                            return None
                        planes.append(gaussian_filter(img[:, :, c], sigma=w_l2, mode='reflect'))
                    blurred = np.stack(planes, axis=-1)
                self._cache_unsharp_blur = (blur_key, blurred)

            if self._is_cancelled(task_id):
                return None

            diff = img - blurred
            if w_denoise > 0.0:
                thresh = np.float32(w_denoise)
                diff = np.sign(diff) * np.maximum(np.abs(diff) - thresh, np.float32(0.0))

            out = img + np.float32(w_l1 - 1.0) * diff
            return np.clip(out, 0.0, 1.0, out=out)

    def _compute_stage2(self, img: np.ndarray, key1, sigma: float,
                        iterations: int, protect_edge: bool, task_id: int):
        """Деконволюция Ричардсона-Люси с пошаговой отменой и дорасчетом итераций из кэша."""
        start_iter = 0
        estimate = None

        if self._cache_rl_core is not None:
            c_key1, c_sigma, c_iters, c_est = self._cache_rl_core
            if c_key1 == key1 and c_sigma == sigma:
                if c_iters == iterations:
                    estimate = c_est
                    start_iter = iterations
                elif c_iters < iterations:
                    # Продолжаем итерации с уже посчитанного приближения
                    estimate = c_est.copy()
                    start_iter = c_iters

        if estimate is None:
            estimate = np.maximum(img.copy(), np.float32(1e-6))

        if start_iter < iterations:
            sigma_axes = (sigma, sigma) if img.ndim == 2 else (sigma, sigma, 0.0)
            for _ in range(start_iter, iterations):
                if self._is_cancelled(task_id):
                    return None
                conv = gaussian_filter(estimate, sigma=sigma_axes, mode='reflect')
                np.maximum(conv, np.float32(1e-12), out=conv)
                relative_blur = img / conv
                if self._is_cancelled(task_id):
                    return None
                corr = gaussian_filter(relative_blur, sigma=sigma_axes, mode='reflect')
                estimate *= corr

            self._cache_rl_core = (key1, sigma, iterations, estimate)

        if self._is_cancelled(task_id):
            return None

        if protect_edge:
            if self._cache_rl_edge_mask is not None and self._cache_rl_edge_mask[0] == key1:
                edge_mask = self._cache_rl_edge_mask[1]
            else:
                if img.ndim == 2:
                    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
                    grad = np.hypot(gx, gy)
                    edge_mask = np.clip(gaussian_filter(grad, sigma=2.0) * 4.0, 0.0, 1.0).astype(np.float32)
                else:
                    masks = []
                    for c in range(img.shape[2]):
                        if self._is_cancelled(task_id):
                            return None
                        plane = img[:, :, c]
                        gx = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
                        gy = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
                        grad = np.hypot(gx, gy)
                        masks.append(np.clip(gaussian_filter(grad, sigma=2.0) * 4.0, 0.0, 1.0).astype(np.float32))
                    edge_mask = np.stack(masks, axis=-1)
                self._cache_rl_edge_mask = (key1, edge_mask)

            weight = edge_mask * np.float32(0.85)
            out = estimate * (np.float32(1.0) - weight) + img * weight
            return np.clip(out, 0.0, 1.0, out=out)

        return np.clip(estimate, 0.0, 1.0)

    def _compute_stage3(self, img: np.ndarray, black_level: float,
                        gamma: float, contrast: float, is_preview: bool, task_id: int):
        """Тональная коррекция и S-Curve (быстрый 16-бит LUT для больших кадров / превью)."""
        if self._is_cancelled(task_id):
            return None

        has_non_linear = abs(gamma - 1.0) > 1e-3 or abs(contrast - 1.0) > 1e-3
        # Для быстрых превью или крупных кадров с нелинейной гаммой/контрастом используем 16-bit LUT
        if has_non_linear and (is_preview or img.size >= 400000):
            lut_in = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
            lut_out = AstroEngine.tone_and_contrast(lut_in, black_level, gamma, contrast).astype(np.float32)
            idx = np.clip(img * np.float32(65535.0), 0.0, 65535.0).astype(np.uint16)
            return lut_out[idx]

        return AstroEngine.tone_and_contrast(
            img,
            black_level=black_level,
            gamma=gamma,
            contrast=contrast
        )
