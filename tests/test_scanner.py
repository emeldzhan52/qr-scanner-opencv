import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from generate_samples import make_qr, perspective, place, textured_background  # noqa: E402
from qr_scanner import QRScanner, parse_payment  # noqa: E402
from qr_scanner.detector import find_finder_patterns, find_qr_candidates  # noqa: E402


@pytest.fixture(scope="module")
def scanner():
    return QRScanner()


def texts(results):
    return sorted(r.text for r in results)


def test_simple_code(scanner):
    assert texts(scanner.scan(make_qr("hello"))) == ["hello"]


def test_cyrillic_text(scanner):
    assert texts(scanner.scan(make_qr("Привет, мир"))) == ["Привет, мир"]


def test_empty_image_returns_nothing(scanner):
    blank = np.full((300, 300, 3), 255, np.uint8)
    assert scanner.scan(blank) == []


@pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 90])
def test_rotated(scanner, angle):
    img = place(textured_background(500, 500), make_qr("rotated", 5), 120, 120, angle)
    assert texts(scanner.scan(img)) == ["rotated"]


def test_multiple_codes(scanner):
    bg = textured_background(720, 1280, seed=1)
    img = place(bg, make_qr("A", 6), 80, 120)
    img = place(img, make_qr("B", 6), 520, 300, angle=25)
    img = place(img, make_qr("C", 5), 950, 80, angle=-15)
    assert texts(scanner.scan(img)) == ["A", "B", "C"]


def test_perspective_and_blur(scanner):
    img = place(textured_background(600, 800, seed=2), make_qr("hard", 4), 260, 140, angle=40)
    img = cv2.GaussianBlur(perspective(img), (3, 3), 0)
    assert texts(scanner.scan(img)) == ["hard"]


def test_finder_patterns_count():
    gray = cv2.cvtColor(make_qr("finder"), cv2.COLOR_BGR2GRAY)
    assert len(find_finder_patterns(gray)) >= 3
    assert len(find_qr_candidates(gray)) == 1


def test_result_points_inside_image(scanner):
    img = make_qr("bbox")
    (res,) = scanner.scan(img)
    h, w = img.shape[:2]
    assert (res.points[:, 0] >= -1).all() and (res.points[:, 0] <= w + 1).all()
    assert (res.points[:, 1] >= -1).all() and (res.points[:, 1] <= h + 1).all()


def test_bad_cascade_path():
    with pytest.raises(ValueError):
        QRScanner(cascade_path="does_not_exist.xml")


# ---------------------------------------------------------------- payment


PAYMENT = (
    "ST00012|Name=Иван Иванов|PersonalAcc=40817810000000000001|BankName=Мини Банк|"
    "BIC=044525000|CorrespAcc=30101810400000000225|Sum=150000|Purpose=Перевод|Extra=1"
)


def test_parse_payment():
    p = parse_payment(PAYMENT)
    assert p is not None
    assert p.name == "Иван Иванов"
    assert p.amount_rub == 1500.0
    assert p.purpose == "Перевод"
    assert p.extra == {"Extra": "1"}


def test_parse_not_payment():
    assert parse_payment("https://example.com") is None


def test_parse_missing_fields():
    assert parse_payment("ST00012|Name=Иван") is None


def test_payment_qr_end_to_end(scanner):
    (res,) = scanner.scan(make_qr(PAYMENT, module_px=6))
    assert parse_payment(res.text).bic == "044525000"
