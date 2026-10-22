"""Сканер QR-кодов на OpenCV с каскадом детекторов."""

from .detector import QRResult, QRScanner
from .payment import PaymentInfo, parse_payment

__all__ = ["QRScanner", "QRResult", "PaymentInfo", "parse_payment"]
