"""Сгенерировать тестовые картинки с QR-кодами в папку samples/.

    python tools/generate_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"

PAYMENT = (
    "ST00012|Name=Иван Иванов|PersonalAcc=40817810000000000001|BankName=Мини Банк|"
    "BIC=044525000|CorrespAcc=30101810400000000225|Sum=150000|Purpose=Перевод по QR"
)


def make_qr(text: str, module_px: int = 8, border: int = 4) -> np.ndarray:
    """QR-код в виде BGR-картинки (только средствами OpenCV)."""
    qr = cv2.QRCodeEncoder.create().encode(text)
    qr = cv2.resize(qr, None, fx=module_px, fy=module_px, interpolation=cv2.INTER_NEAREST)
    pad = border * module_px
    qr = cv2.copyMakeBorder(qr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)


def textured_background(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Фон «как на фото»: градиент + шум, чтобы задача была не тривиальной."""
    rng = np.random.default_rng(seed)
    grad = np.linspace(90, 200, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    bg = np.dstack([grad * 0.9, grad, grad * 1.1])
    bg += rng.normal(0, 12, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def place(bg: np.ndarray, qr: np.ndarray, x: int, y: int, angle: float = 0.0) -> np.ndarray:
    """Вклеить QR в фон с поворотом."""
    h, w = qr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    rotated = cv2.warpAffine(qr, m, (nw, nh), borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), m, (nw, nh))

    out = bg.copy()
    roi = out[y : y + nh, x : x + nw]
    roi[mask > 0] = rotated[mask > 0]
    return out


def perspective(img: np.ndarray, strength: float = 0.08) -> np.ndarray:
    h, w = img.shape[:2]
    d = strength * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d, d * 0.5], [w - d * 0.3, 0], [w, h], [0, h - d]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderMode=cv2.BORDER_REPLICATE)


def main() -> int:
    OUT.mkdir(exist_ok=True)

    cv2.imwrite(str(OUT / "simple.png"), make_qr("https://gitlab.com"))
    cv2.imwrite(str(OUT / "payment.png"), make_qr(PAYMENT, module_px=6))

    bg = textured_background(720, 1280, seed=1)
    img = place(bg, make_qr("Код №1", 6), 80, 120, angle=0)
    img = place(img, make_qr("Код №2", 6), 520, 300, angle=25)
    img = place(img, make_qr("Код №3", 5), 950, 80, angle=-15)
    cv2.imwrite(str(OUT / "multiple.png"), img)

    bg = textured_background(600, 800, seed=2)
    hard = place(bg, make_qr("Повёрнутый и наклонённый код", 4), 260, 140, angle=40)
    hard = perspective(hard)
    hard = cv2.GaussianBlur(hard, (3, 3), 0)
    cv2.imwrite(str(OUT / "hard_rotated.png"), hard)

    dark = (make_qr("Тёмный кадр", 6).astype(np.float32) * 0.35 + 20).astype(np.uint8)
    cv2.imwrite(str(OUT / "low_contrast.png"), dark)

    print(f"Готово: {sorted(p.name for p in OUT.glob('*.png'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
