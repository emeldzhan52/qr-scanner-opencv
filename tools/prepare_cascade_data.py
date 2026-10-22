"""Подготовить датасет для обучения Haar/LBP-каскада, находящего QR-коды.

Создаёт:
    cascade_data/pos/*.png    — QR-коды со случайными искажениями
    cascade_data/neg/*.png    — фоны без QR-кодов
    cascade_data/positives.txt и negatives.txt — списки для утилит OpenCV

    python tools/prepare_cascade_data.py --pos 1000 --neg 2000

Само обучение делается утилитами opencv_createsamples и opencv_traincascade
(см. раздел «Обучение каскада» в README).
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import cv2
import numpy as np

from generate_samples import make_qr, textured_background

ROOT = Path(__file__).resolve().parent.parent
SIZE = 24  # размер окна каскада, w = h = 24


def random_text(rng: random.Random) -> str:
    n = rng.randint(5, 60)
    return "".join(rng.choices(string.ascii_letters + string.digits, k=n))


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Случайные поворот на кратный 90° угол, яркость, шум и размытие."""
    img = np.rot90(img, rng.randint(0, 3)).copy()
    alpha = rng.uniform(0.6, 1.2)
    beta = rng.uniform(-40, 40)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(0, 12), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def make_negative(rng: random.Random, seed: int) -> np.ndarray:
    """Фон без QR: градиент + случайные прямоугольники и линии (ловушки для детектора)."""
    h, w = 240, 320
    img = cv2.cvtColor(textured_background(h, w, seed=seed), cv2.COLOR_BGR2GRAY)
    for _ in range(rng.randint(3, 15)):
        x, y = rng.randint(0, w), rng.randint(0, h)
        color = rng.randint(0, 255)
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + rng.randint(5, 80), y + rng.randint(5, 80)), color, -1)
        else:
            cv2.line(img, (x, y), (rng.randint(0, w), rng.randint(0, h)), color, rng.randint(1, 4))
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=int, default=1000)
    parser.add_argument("--neg", type=int, default=2000)
    parser.add_argument("--out", default=str(ROOT / "cascade_data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "pos").mkdir(parents=True, exist_ok=True)
    (out / "neg").mkdir(parents=True, exist_ok=True)

    pos_lines = []
    for i in range(args.pos):
        qr = cv2.cvtColor(make_qr(random_text(rng), module_px=4, border=1), cv2.COLOR_BGR2GRAY)
        qr = augment(qr, rng)
        name = f"pos/{i:05d}.png"
        cv2.imwrite(str(out / name), qr)
        h, w = qr.shape
        pos_lines.append(f"{name} 1 0 0 {w} {h}")  # формат: путь кол-во x y w h

    neg_lines = []
    for i in range(args.neg):
        name = f"neg/{i:05d}.png"
        cv2.imwrite(str(out / name), make_negative(rng, seed=args.seed + i))
        neg_lines.append(name)

    (out / "positives.txt").write_text("\n".join(pos_lines) + "\n")
    (out / "negatives.txt").write_text("\n".join(neg_lines) + "\n")
    print(f"Позитивов: {args.pos}, негативов: {args.neg} -> {out}")
    print(f"Размер окна каскада: {SIZE}x{SIZE}")


if __name__ == "__main__":
    main()
