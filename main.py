# AstroRack DAW v8.3.6 — Lunar, Planetary & Deep-Sky Image Mastering Platform
# Точка входа в приложение.

import os
import sys
import ctypes
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from utils.file_io import get_resource_path
from ui.splash import AstroSplashScreen
from ui.main_window import AstroRackWindow


def main():
    # 0. Регистрация AppUserModelID для Windows, чтобы иконка закреплялась в таскбаре
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('astrorack.daw.processor.v8.3.6')
    except Exception:
        pass

    # Синхронизируем рабочий каталог в режиме frozen bundle для корректной работы относительных ассетов в QSS
    if hasattr(sys, '_MEIPASS'):
        try:
            os.chdir(sys._MEIPASS)
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Установка иконки для всего приложения
    ico_path = get_resource_path(os.path.join("assets", "app_icon.ico"))
    icon_256 = get_resource_path(os.path.join("assets", "icons", "icon_256x256.png"))
    if os.path.exists(icon_256):
        app.setWindowIcon(QIcon(icon_256))
    elif os.path.exists(ico_path):
        app.setWindowIcon(QIcon(ico_path))
    else:
        app.setWindowIcon(QIcon(get_resource_path(os.path.join("assets", "icons", "icon_256x256.png"))))

    # 1. Показываем стильный сплэш-скрин при запуске
    splash = AstroSplashScreen()
    splash.show()
    splash.set_progress(15, "Initializing DSP core & SIMD routines...")
    app.processEvents()

    holder = {"window": None}

    def step1():
        splash.set_progress(40, "Loading DAW rack modules & wavelet engines...")
        QTimer.singleShot(300, step2)

    def step2():
        splash.set_progress(65, "Calibrating GPU acceleration & pipeline cache...")
        QTimer.singleShot(300, step3)

    def step3():
        splash.set_progress(85, "Building UI & canvas workspace...")
        holder["window"] = AstroRackWindow()
        QTimer.singleShot(350, step4)

    def step4():
        splash.set_progress(100, "Ready")
        QTimer.singleShot(200, finish)

    def finish():
        splash.fade_out_and_finish(holder["window"], duration_ms=350)

    # Запуск этапов инициализации (общая продолжительность ~1.3–1.4 секунды)
    QTimer.singleShot(250, step1)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
