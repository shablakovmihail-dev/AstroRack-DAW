#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroRack DAW — Генерация пакета иконок приложения из прозрачного PNG.
Открывает assets/logo_raw.png (RGBA), генерирует чистые иконки PNG и мультиформатный ICO.
"""

import os
import sys
import numpy as np
from PIL import Image

SIZES = [16, 24, 32, 48, 64, 128, 256, 512]
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def get_project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def build_icons(raw_path: str, output_dir: str, ico_path: str):
    print(f"[1/3] Загрузка исходного логотипа: {raw_path}")
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Файл {raw_path} не найден!")

    img = Image.open(raw_path).convert("RGBA")

    # Проверка наличия альфа-канала
    if "A" not in img.getbands():
        raise ValueError("Ошибка: у изображения отсутствует альфа-канал прозрачности!")

    # Гарантируем чистый 0-альфа фон без остаточного микро-шума экспорта
    arr = np.array(img)
    arr[:, :, 3] = np.where(arr[:, :, 3] < 6, 0, arr[:, :, 3])
    clean_img = Image.fromarray(arr, mode="RGBA")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[2/3] Генерация чистых PNG с ресемплингом Image.Resampling.LANCZOS:")
    png_images = {}
    for size in SIZES:
        resized = clean_img.resize((size, size), resample=Image.Resampling.LANCZOS)
        out_filename = f"icon_{size}x{size}.png"
        out_path = os.path.join(output_dir, out_filename)
        resized.save(out_path, format="PNG", optimize=True)
        png_images[size] = resized
        print(f"  ✓ {out_filename:20} -> {out_path}")

    # Сборка мультиформатного Windows ICO
    print(f"[3/3] Сборка мультиформатного Windows ICO: {ico_path}")
    ico_frames = [png_images[s] for s in ICO_SIZES if s in png_images]
    if ico_frames:
        base_frame = png_images.get(256, ico_frames[-1])
        base_frame.save(
            ico_path,
            format="ICO",
            sizes=[(s, s) for s in ICO_SIZES],
            append_images=ico_frames
        )
        ico_size_kb = os.path.getsize(ico_path) / 1024.0
        print(f"  ✓ {os.path.basename(ico_path)} успешно создан ({ico_size_kb:.1f} KB, слои: {ICO_SIZES})")

    print("\n>>> Генерация пакета иконок завершена успешно! <<<")


def main():
    root = get_project_root()
    raw_path = os.path.join(root, "assets", "logo_raw.png")
    output_dir = os.path.join(root, "assets", "icons")
    ico_path = os.path.join(root, "assets", "app_icon.ico")

    build_icons(raw_path, output_dir, ico_path)


if __name__ == "__main__":
    main()
