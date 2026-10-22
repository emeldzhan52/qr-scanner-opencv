"""Сравнение: обычный cv2.QRCodeDetector против каскада QRScanner.

Генерирует сцены с кодами под разными углами, с перспективой, размытием
и несколькими кодами в кадре, и считает долю правильно прочитанных кодов.

    python tools/benchmark.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generate_samples import make_qr, perspective, place, textured_background  # noqa: E402
from qr_scanner import QRScanner  # noqa: E402


def baseline(img) -> set[str]:
    ok, texts, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(img)
    return {t for t in texts if t} if ok else set()


def make_scenes(n: int, seed: int = 0):
    rng = random.Random(seed)
    for i in range(n):
        bg = textured_background(720, 1280, seed=i)
        k = rng.choice([1, 1, 2, 3])
        expected = set()
        img = bg
        for j in range(k):
            text = f"scene{i}-code{j}"
            expected.add(text)
            module = rng.choice([4, 5, 6])
            x = 60 + j * 400 + rng.randint(0, 60)
            y = rng.randint(40, 300)
            img = place(img, make_qr(text, module), x, y, angle=rng.uniform(0, 90))
        if rng.random() < 0.4:
            img = perspective(img, strength=rng.uniform(0.03, 0.1))
        if rng.random() < 0.4:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        yield img, expected


def main() -> None:
    scenes = list(make_scenes(60))
    total = sum(len(e) for _, e in scenes)
    scanner = QRScanner()

    for name, fn in [
        ("QRCodeDetector", baseline),
        ("QRScanner (каскад)", lambda im: {r.text for r in scanner.scan(im)}),
    ]:
        hits, t0 = 0, time.perf_counter()
        for img, expected in scenes:
            hits += len(fn(img) & expected)
        ms = (time.perf_counter() - t0) * 1000 / len(scenes)
        print(f"{name:<22} распознано {hits:>3}/{total} ({hits / total:6.1%}), {ms:6.0f} мс/кадр")


if __name__ == "__main__":
    main()
