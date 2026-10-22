"""Точка входа: сканирование QR-кодов с картинки или веб-камеры.

Примеры:
    python main.py --image samples/payment.png
    python main.py --image samples/hard_rotated.png --save out.png
    python main.py --camera 0
    python main.py --camera 0 --cascade cascades/qr_cascade.xml
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from qr_scanner import QRScanner, parse_payment
from qr_scanner.detector import draw_results


def print_result(res) -> None:
    print(f"[{res.method}] {res.text}")
    payment = parse_payment(res.text)
    if payment:
        print("  Платёжный QR:")
        for line in payment.summary().splitlines():
            print("   ", line)


def run_image(scanner: QRScanner, path: str, save: str | None, show: bool) -> int:
    frame = cv2.imread(path)
    if frame is None:
        print(f"Не удалось открыть {path}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    results = scanner.scan(frame)
    elapsed = (time.perf_counter() - start) * 1000

    if not results:
        print(f"QR-коды не найдены ({elapsed:.0f} мс)")
    else:
        print(f"Найдено кодов: {len(results)} ({elapsed:.0f} мс)")
        for res in results:
            print_result(res)

    annotated = draw_results(frame, results)
    if save:
        cv2.imwrite(save, annotated)
        print(f"Результат сохранён в {save}")
    if show:
        cv2.imshow("QR scanner", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_camera(scanner: QRScanner, index: int) -> int:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Камера {index} недоступна", file=sys.stderr)
        return 1

    seen: set[str] = set()
    print("Нажмите q, чтобы выйти")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = scanner.scan(frame)
        for res in results:
            if res.text not in seen:  # печатаем каждый код один раз
                seen.add(res.text)
                print_result(res)
        cv2.imshow("QR scanner", draw_results(frame, results))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Сканер QR-кодов на OpenCV")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="путь к изображению")
    src.add_argument("--camera", type=int, help="номер веб-камеры (обычно 0)")
    parser.add_argument("--cascade", help="путь к обученному Haar/LBP-каскаду (.xml)")
    parser.add_argument("--no-contours", action="store_true", help="отключить поиск по контурам")
    parser.add_argument("--save", help="сохранить картинку с разметкой")
    parser.add_argument("--show", action="store_true", help="показать окно с результатом")
    args = parser.parse_args()

    scanner = QRScanner(cascade_path=args.cascade, use_contour_search=not args.no_contours)
    if args.image:
        return run_image(scanner, args.image, args.save, args.show)
    return run_camera(scanner, args.camera)


if __name__ == "__main__":
    sys.exit(main())
