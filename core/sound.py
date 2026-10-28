"""Тактильный звуковой движок AstroRack DAW (Subtle Audio Engine).
Синтезирует и воспроизводит мягкие аналоговые микро-звуки для ключевых действий интерфейса:
- Клик Master Bypass: аналоговый студийный щелчок реле компрессора (~140-360 Гц с механическим контактом).
- Успешный экспорт: мягкий звук затвора камеры/линзы.
"""
import os
import io
import wave
import threading
import numpy as np
from PyQt6.QtCore import QSettings
from utils.file_io import get_resource_path

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    winsound = None
    _HAS_WINSOUND = False


class SoundEngine:
    """Служба тактильных звуковых эффектов с поддержкой сохранения состояния Mute."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

        self.settings = QSettings("AstroRack", "Audio")
        # По умолчанию звук включен (sound_muted = False)
        self._enabled = self.settings.value("ui_sounds_enabled", True, type=bool)

        self._sounds_dir = get_resource_path(os.path.join("assets", "sounds"))
        os.makedirs(self._sounds_dir, exist_ok=True)

        self._relay_path = os.path.join(self._sounds_dir, "relay_click.wav")
        self._shutter_path = os.path.join(self._sounds_dir, "camera_shutter.wav")

        self._ensure_sound_files()

        # Предзагрузка байтов в память для мгновенного отклика
        self._relay_bytes = b""
        self._shutter_bytes = b""
        if os.path.isfile(self._relay_path):
            try:
                with open(self._relay_path, "rb") as f:
                    self._relay_bytes = f.read()
            except Exception:
                pass
        if os.path.isfile(self._shutter_path):
            try:
                with open(self._shutter_path, "rb") as f:
                    self._shutter_bytes = f.read()
            except Exception:
                pass

        # Инициализация QSoundEffect как постоянных атрибутов класса
        self._fx_relay = None
        self._fx_shutter = None
        try:
            from PyQt6.QtMultimedia import QSoundEffect
            from PyQt6.QtCore import QUrl
            if os.path.isfile(self._relay_path):
                self._fx_relay = QSoundEffect()
                self._fx_relay.setSource(QUrl.fromLocalFile(os.path.abspath(self._relay_path)))
                self._fx_relay.setVolume(1.0)
            if os.path.isfile(self._shutter_path):
                self._fx_shutter = QSoundEffect()
                self._fx_shutter.setSource(QUrl.fromLocalFile(os.path.abspath(self._shutter_path)))
                self._fx_shutter.setVolume(1.0)
        except Exception:
            pass

    def _ensure_sound_files(self):
        """Генерирует файлы коротких эффектов высокого качества."""
        sr = 44100
        # 1. Аналоговый щелчок студийного реле компрессора (35 мс, 1400 Гц + 360 Гц + 160 Гц)
        if not os.path.exists(self._relay_path) or os.path.getsize(self._relay_path) < 1000:
            dur1 = 0.035
            t1 = np.linspace(0, dur1, int(sr * dur1), endpoint=False)
            tr1 = np.sin(2 * np.pi * 1400 * t1) * np.exp(-t1 / 0.002) * 0.45
            bd1 = np.sin(2 * np.pi * 360 * t1) * np.exp(-t1 / 0.008) * 0.55
            th1 = np.sin(2 * np.pi * 160 * t1) * np.exp(-t1 / 0.014) * 0.40
            t_b = t1 - 0.007
            bn1 = np.where(t_b >= 0, np.sin(2 * np.pi * 1100 * t_b) * np.exp(-t_b / 0.003) * 0.30, 0.0)
            sig1 = tr1 + bd1 + th1 + bn1
            sig1 = (sig1 / max(1e-6, np.max(np.abs(sig1)))) * 0.88
            fade_len = int(sr * 0.001)
            sig1[:fade_len] *= np.linspace(0, 1, fade_len)
            sig1[-fade_len:] *= np.linspace(1, 0, fade_len)
            arr1 = (sig1 * 32767).astype(np.int16)
            try:
                with wave.open(self._relay_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(arr1.tobytes())
            except Exception:
                pass

        # 2. Мягкий затвор камеры/линзы (45 мс, две шторки + текстура)
        if not os.path.exists(self._shutter_path) or os.path.getsize(self._shutter_path) < 1000:
            dur2 = 0.045
            t2 = np.linspace(0, dur2, int(sr * dur2), endpoint=False)
            c1 = np.sin(2 * np.pi * 2200 * t2) * np.exp(-t2 / 0.003) * 0.45
            b1 = np.sin(2 * np.pi * 520 * t2) * np.exp(-t2 / 0.007) * 0.35
            t_c2 = t2 - 0.016
            c2 = np.where(t_c2 >= 0, np.sin(2 * np.pi * 1750 * t_c2) * np.exp(-t_c2 / 0.003) * 0.50, 0.0)
            b2 = np.where(t_c2 >= 0, np.sin(2 * np.pi * 380 * t_c2) * np.exp(-t_c2 / 0.008) * 0.40, 0.0)
            noise = np.random.RandomState(42).normal(0, 0.08, len(t2)) * (np.exp(-t2 / 0.006) + np.where(t_c2 >= 0, np.exp(-t_c2 / 0.006), 0.0))
            sig2 = c1 + b1 + c2 + b2 + noise
            sig2 = (sig2 / max(1e-6, np.max(np.abs(sig2)))) * 0.88
            fade_len = int(sr * 0.001)
            sig2[:fade_len] *= np.linspace(0, 1, fade_len)
            sig2[-fade_len:] *= np.linspace(1, 0, fade_len)
            arr2 = (sig2 * 32767).astype(np.int16)
            try:
                with wave.open(self._shutter_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(arr2.tobytes())
            except Exception:
                pass

    @property
    def sound_muted(self) -> bool:
        """Показывает, отключен ли звук (True = без звука)."""
        return not self._enabled

    @sound_muted.setter
    def sound_muted(self, val: bool):
        self.set_enabled(not val)

    def is_enabled(self) -> bool:
        """Возвращает текущее состояние активности звуковых эффектов."""
        return self._enabled

    def set_enabled(self, enabled: bool):
        """Устанавливает и сохраняет активность звуковых эффектов."""
        self._enabled = bool(enabled)
        self.settings.setValue("ui_sounds_enabled", self._enabled)

    def toggle_enabled(self) -> bool:
        """Переключает режим активности звуков (Mute / Unmute)."""
        self.set_enabled(not self._enabled)
        return self._enabled

    def _play(self, wav_bytes: bytes, wav_path: str, qsound_effect=None):
        """Безопасное асинхронное воспроизведение звука."""
        if self.sound_muted:
            return

        # 1. Приоритетный способ под Windows — стандартный модуль winsound
        played = False
        if _HAS_WINSOUND and winsound is not None:
            try:
                winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                played = True
            except Exception:
                try:
                    if wav_bytes:
                        threading.Thread(target=winsound.PlaySound, args=(wav_bytes, winsound.SND_MEMORY), daemon=True).start()
                        played = True
                except Exception:
                    pass

        # 2. QSoundEffect как дополнение/альтернатива
        if not played and qsound_effect is not None:
            try:
                qsound_effect.play()
            except Exception:
                pass

    def play_relay_click(self):
        """Воспроизводит аналоговый щелчок студийного реле компрессора (Master Bypass)."""
        print("[Sound] Playing relay click...")
        self._play(self._relay_bytes, self._relay_path, self._fx_relay)

    def play_shutter(self):
        """Воспроизводит мягкий звук затвора камеры/линзы (успешный экспорт)."""
        print("[Sound] Playing camera shutter...")
        self._play(self._shutter_bytes, self._shutter_path, self._fx_shutter)


sound_engine = SoundEngine()
