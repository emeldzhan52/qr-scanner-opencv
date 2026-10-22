"""Поиск и распознавание QR-кодов.

Распознавание устроено как каскад: дешёвые этапы идут первыми, дорогие
запускаются только если предыдущие ничего не нашли.

    1. QRCodeDetector по всему кадру            (быстро, работает в простых случаях)
    2. Поиск кандидатов:
         - Haar/LBP-каскад, если передан .xml   (cv2.CascadeClassifier)
         - поиск «глазков» QR по контурам       (finder patterns, без обучения)
    3. Декодирование каждого кандидата (ROI) с разной предобработкой:
         оригинал -> выравнивание контраста (CLAHE) -> бинаризация Оцу -> увеличение
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class QRResult:
    text: str
    points: np.ndarray  # 4 угла кода в координатах исходного кадра, shape (4, 2)
    method: str  # каким этапом каскада найден код
    bbox: tuple[int, int, int, int] = field(init=False)  # x, y, w, h

    def __post_init__(self) -> None:
        pts = self.points.astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        self.bbox = (int(x), int(y), int(w), int(h))


class QRScanner:
    def __init__(
        self,
        cascade_path: str | Path | None = None,
        use_contour_search: bool = True,
        roi_padding: float = 0.25,
    ) -> None:
        self.detector = cv2.QRCodeDetector()
        self.use_contour_search = use_contour_search
        self.roi_padding = roi_padding

        self.cascade: cv2.CascadeClassifier | None = None
        if cascade_path is not None:
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if cascade.empty():
                raise ValueError(f"Не удалось загрузить каскад: {cascade_path}")
            self.cascade = cascade

    # ------------------------------------------------------------------ public

    def scan(self, frame: np.ndarray) -> list[QRResult]:
        """Найти и распознать все QR-коды на кадре (BGR или grayscale)."""
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Этап 1: весь кадр целиком.
        found: dict[str, QRResult] = {
            r.text: r for r in self._decode_multi(gray, offset=(0, 0), method="full-frame")
        }

        # Этап 2: кандидаты.
        candidates: list[tuple[tuple[int, int, int, int], str]] = []
        if self.cascade is not None:
            candidates += [(box, "cascade") for box in self._cascade_candidates(gray)]
        if self.use_contour_search:
            candidates += [(box, "contours") for box in find_qr_candidates(gray)]

        # Этап 3: декодируем только те области, где код ещё не прочитан.
        for box, source in candidates:
            if any(_overlap(box, r.bbox) > 0.5 for r in found.values()):
                continue
            for res in self._decode_roi(gray, box, source):
                found.setdefault(res.text, res)
        return list(found.values())

    # ----------------------------------------------------------------- helpers

    def _decode_multi(
        self, gray: np.ndarray, offset: tuple[int, int], method: str, scale: float = 1.0
    ) -> list[QRResult]:
        try:
            ok, texts, points, _ = self.detector.detectAndDecodeMulti(gray)
        except cv2.error:
            return []
        if not ok or points is None:
            return []
        results = []
        for text, pts in zip(texts, points):
            if not text:
                continue  # код найден, но не прочитан
            pts = pts.reshape(4, 2) / scale + np.array(offset, dtype=np.float32)
            results.append(QRResult(text=text, points=pts, method=method))
        return results

    def _decode_roi(
        self, gray: np.ndarray, box: tuple[int, int, int, int], source: str
    ) -> list[QRResult]:
        x, y, w, h = _pad_box(box, self.roi_padding, gray.shape)
        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            return []

        for name, img, scale in _preprocess_variants(roi):
            res = self._decode_multi(img, offset=(x, y), method=f"{source}+{name}", scale=scale)
            if res:
                return res
        return []

    def _cascade_candidates(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        assert self.cascade is not None
        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        return [tuple(int(v) for v in b) for b in boxes]


# ---------------------------------------------------------------------------
# Поиск кандидатов по контурам (без обучения)
# ---------------------------------------------------------------------------


def find_finder_patterns(gray: np.ndarray, min_size: int = 7) -> list[np.ndarray]:
    """Найти «глазки» QR-кода: квадрат в квадрате в квадрате.

    Используем иерархию контуров: у finder pattern внешний контур содержит
    вложенный, а тот — ещё один (соотношение сторон 7:5:3).
    """
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7
    )
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    patterns = []
    for i, cnt in enumerate(contours):
        # Глубина вложенности: считаем, сколько раз можно спуститься к ребёнку.
        depth, child = 0, hierarchy[i][2]
        while child != -1:
            depth += 1
            child = hierarchy[child][2]
        if depth < 2:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_size or h < min_size:
            continue
        if not 0.6 < w / h < 1.6:  # должен быть примерно квадратным
            continue
        approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
        if len(approx) != 4:
            continue
        patterns.append(cnt)
    return patterns


def _drop_containers(patterns: list[np.ndarray]) -> list[np.ndarray]:
    """Убрать контуры, внутри которых лежат другие кандидаты.

    Рамка всего QR-кода тоже имеет глубокую вложенность и квадратную форму,
    но глазком не является — внутри неё лежат настоящие глазки.
    """
    rects = [cv2.boundingRect(c) for c in patterns]
    keep = []
    for i, (x, y, w, h) in enumerate(rects):
        contains_other = any(
            j != i and x < ox + ow / 2 < x + w and y < oy + oh / 2 < y + h and ow < w * 0.6
            for j, (ox, oy, ow, oh) in enumerate(rects)
        )
        if not contains_other:
            keep.append(patterns[i])
    return keep


def find_qr_candidates(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Сгруппировать «глазки» по три — это и есть вероятные QR-коды.

    Три глазка одного кода лежат в вершинах прямоугольного равнобедренного
    треугольника и имеют одинаковый размер. Проверяем каждую тройку.
    """
    patterns = _drop_containers(find_finder_patterns(gray))[:40]  # защита от O(n^3)
    if len(patterns) < 3:
        return []

    rects = [cv2.boundingRect(c) for c in patterns]
    centers = np.array([(x + w / 2, y + h / 2) for x, y, w, h in rects])
    # Сторона глазка через площадь: в отличие от boundingRect не зависит от поворота.
    sizes = np.array([np.sqrt(cv2.contourArea(c)) for c in patterns], dtype=np.float32)

    scored = []
    n = len(patterns)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx = (i, j, k)
                s = sizes[list(idx)]
                if s.max() / s.min() > 1.5:  # глазки разного размера
                    continue
                a, b, c = sorted(
                    [
                        np.linalg.norm(centers[i] - centers[j]),
                        np.linalg.norm(centers[j] - centers[k]),
                        np.linalg.norm(centers[i] - centers[k]),
                    ]
                )
                if a < 1.5 * s.mean() or a > 12 * s.mean():
                    continue
                legs_ratio = a / b  # катеты равны -> 1.0
                hyp_error = abs(c - np.hypot(a, b)) / c  # теорема Пифагора
                if legs_ratio < 0.8 or hyp_error > 0.12:
                    continue
                scored.append((hyp_error + (1 - legs_ratio), idx))

    # Жадно берём лучшие тройки, не используя один глазок дважды.
    used: set[int] = set()
    boxes = []
    for _, idx in sorted(scored):
        if used & set(idx):
            continue
        used |= set(idx)
        # Угол A — вершина прямого угла (напротив самой длинной стороны).
        # Четвёртый угол кода (где глазка нет) достраиваем: D = B + C - A.
        i, j, k = idx
        others = {i: (j, k), j: (i, k), k: (i, j)}

        def opposite_side(t: int) -> float:
            p, q = others[t]
            return float(np.linalg.norm(centers[p] - centers[q]))

        a_idx = max(idx, key=opposite_side)
        b_idx, c_idx = others[a_idx]
        d = centers[b_idx] + centers[c_idx] - centers[a_idx]
        half = sizes[list(idx)].mean()
        corner = np.array([d - half, d + half], dtype=np.float32)
        pts = np.vstack([patterns[t].reshape(-1, 2).astype(np.float32) for t in idx] + [corner])
        boxes.append(tuple(int(v) for v in cv2.boundingRect(pts)))
    return boxes


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def _overlap(a, b) -> float:
    """Какая доля меньшего прямоугольника покрыта пересечением."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    smaller = min(aw * ah, bw * bh)
    return (iw * ih) / smaller if smaller else 0.0


def _pad_box(box, padding, shape):
    x, y, w, h = box
    dx, dy = int(w * padding), int(h * padding)
    H, W = shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(W, x + w + dx), min(H, y + h + dy)
    return x0, y0, x1 - x0, y1 - y0


def _preprocess_variants(roi: np.ndarray):
    """Генератор вариантов предобработки: (название, изображение, масштаб)."""
    yield "raw", roi, 1.0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)
    yield "clahe", clahe, 1.0

    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu", otsu, 1.0

    if max(roi.shape) < 400:  # маленький код — увеличим
        scale = 400 / max(roi.shape)
        big = cv2.resize(otsu, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        yield "upscale", big, scale


def draw_results(frame: np.ndarray, results: list[QRResult]) -> np.ndarray:
    """Нарисовать рамки и текст найденных кодов."""
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for res in results:
        pts = res.points.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 200, 0), thickness=3)
        x, y, _, _ = res.bbox
        label = res.text if len(res.text) <= 40 else res.text[:37] + "..."
        cv2.putText(
            out, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 255), 2, cv2.LINE_AA,
        )
    return out
